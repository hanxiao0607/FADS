import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

class LogDataset(Dataset):
    def __init__(self, seqs, ys):
        super().__init__()
        self.seqs = seqs
        self.ys = ys

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.seqs[idx],  self.ys[idx]

def get_iter(X, y, batch_size=1024, shuffle=False):
    if y is None:
        y = [-1 for _ in range(len(X))]
    dataset = LogDataset(X, y)
    if shuffle == True:
        iter = DataLoader(dataset, batch_size, shuffle=True, worker_init_fn=np.random.seed(42))
    else:
        iter = DataLoader(dataset, batch_size, worker_init_fn=np.random.seed(42))
    return iter

class MLP(object):
    def __init__(self, input_dim, output_dim, batch_size=1024, max_epoch=50, device='cuda:1'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = device
        self.max_epoch = max_epoch

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.output_dim)

        ).to(self.device)

        self.optim = optim.Adam(self.net.parameters())
        self.loss_CEL = nn.CrossEntropyLoss()

    def _train(self, iterator):
        self.net.train()

        epoch_loss = 0

        for batch in iterator:
            src = batch[0].to(self.device)
            label = batch[1].to(self.device)
            self.optim.zero_grad()
            output = self.net(src)
            loss = self.loss_CEL(output, label)
            loss.backward()

            self.optim.step()
            epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    def _evaluate(self, iterator):
        self.net.eval()
        epoch_loss = 0
        lst_out = []
        with torch.no_grad():
            for batch in iterator:
                src = batch[0].to(self.device)
                label = batch[1].to(self.device)
                output = self.net(src)
                _, predicted = torch.max(output, 1)
                lst_out.extend(predicted.detach().cpu().tolist())
                loss = self.loss_CEL(output, label)
                epoch_loss += loss.item()
        return epoch_loss / len(iterator), lst_out

    def train_MLP(self, train_x, train_y):
        x_train, x_eval, y_train, y_eval = train_test_split(train_x, train_y, test_size=0.4, random_state=42)
        train_iter = get_iter(torch.tensor(x_train).float(), torch.tensor(y_train).long(), self.batch_size)
        eval_iter = get_iter(torch.tensor(x_eval).float(), torch.tensor(y_eval).long(), self.batch_size)

        best_eval_loss = float('inf')

        for epoch in range(self.max_epoch):
            train_loss = self._train(train_iter)
            eval_loss, _ = self._evaluate(eval_iter)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(self.net.state_dict(), './saved_models/MLP.pt')

    def load_model(self):
        self.net.load_state_dict(torch.load('./saved_models/MLP.pt'))
        self.net.to(self.device)

    def predict(self, test_x, text_y):
        self.load_model()
        self.net.eval()
        test_iter = get_iter(torch.tensor(test_x).float(), torch.tensor(text_y).long(), self.batch_size, shuffle=False)
        _, lst_pred = self._evaluate(test_iter)
        return lst_pred
