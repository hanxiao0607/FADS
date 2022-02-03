import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

class LogDataset(Dataset):
    def __init__(self, seqs, lens, ys):
        super().__init__()
        self.seqs = seqs
        self.lens = lens
        self.ys = ys

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.seqs[idx], self.lens[idx], self.ys[idx]


def list_to_tensor(lst):
    if isinstance(lst[0][0], list):
        lst = [i[0] for i in lst]
    x_lens = [len(i) for i in lst]
    max_len = max(x_lens)
    for i in range(len(lst)):
        if len(lst[i]) < max_len:
            lst[i].extend([0 for _ in range(max_len - len(lst[i]))])
    lst = torch.stack([torch.tensor(i) for i in lst])
    return lst, x_lens


def get_iter(x_and_len, y, batch_size=1024, shuffle=False):
    X, lens = x_and_len
    if y is None:
        y = [-1 for _ in range(len(X))]
    dataset = LogDataset(X, lens, y)
    if shuffle == True:
        iter = DataLoader(dataset, batch_size, shuffle=True, worker_init_fn=np.random.seed(42))
    else:
        iter = DataLoader(dataset, batch_size, worker_init_fn=np.random.seed(42))
    return iter


class classification_lstm(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, out_dim):
        super(classification_lstm, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid_dim, 1, bidirectional=False, batch_first=True)
        self.fc1 = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input, lens):
        embedded = self.dropout(self.embedding(input))
        output, _ = self.rnn(embedded)
        prediction = [self.fc1(torch.mean(output[i, :lens[i], :], dim=0)) for i in range(len(lens))]
        prediction = torch.stack(prediction)
        return prediction


class MLP(object):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, batch_size=1024, max_epoch=50, device='cuda:1'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.batch_size = batch_size
        self.device = device
        self.max_epoch = max_epoch

        self.net = classification_lstm(self.input_dim, self.emb_dim, self.hid_dim, self.output_dim).to(self.device).to(self.device)

        self.optim = optim.Adam(self.net.parameters())
        self.loss_CEL = nn.CrossEntropyLoss()

    def _train(self, iterator):
        self.net.train()

        epoch_loss = 0

        for batch in iterator:
            src = batch[0].to(self.device)
            lens = batch[1]
            label = batch[2].to(self.device)
            self.optim.zero_grad()
            output = self.net(src, lens)
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
                lens = batch[1]
                label = batch[2].to(self.device)
                output = self.net(src, lens)
                _, predicted = torch.max(output, 1)
                lst_out.extend(predicted.detach().cpu().tolist())
                loss = self.loss_CEL(output, label)
                epoch_loss += loss.item()
        return epoch_loss / len(iterator), lst_out

    def train_MLP(self, train_x, train_y):
        x_train, x_eval, y_train, y_eval = train_test_split(train_x, train_y, test_size=0.4, random_state=42)
        train_iter = get_iter(list_to_tensor(x_train), torch.tensor(y_train).long(), self.batch_size)
        eval_iter = get_iter(list_to_tensor(x_eval), torch.tensor(y_eval).long(), self.batch_size)

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
        test_iter = get_iter(list_to_tensor(test_x), torch.tensor(text_y).long(), self.batch_size, shuffle=False)
        _, lst_pred = self._evaluate(test_iter)
        return lst_pred
