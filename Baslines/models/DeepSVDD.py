import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def get_dist(ts, center):
    ts = ts.cpu().detach().numpy()
    center = center.cpu().numpy()
    temp = []
    for i in ts:
        temp.append(np.linalg.norm(i-center))
    return temp

def get_center(emb):
    return torch.mean(emb, 0)

class PointDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx]


class DeepSVDD(object):

    def __init__(self, input_dim, out_dim, batch_size=1024, eps=0.1, max_epoch=50, nu=0.05, device='cuda:1'):
        super().__init__()

        self.nu = nu
        self.R = 0.0
        self.c = None
        self.eps = eps
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.device = device
        self.batch_size = batch_size
        self.max_epoch = max_epoch

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.out_dim)

        ).to(self.device)

        self.optim = optim.Adam(self.net.parameters())
        self.loss_mse = nn.MSELoss()

    def _train(self, iterator, center):
        self.net.train()

        epoch_loss = 0

        for (i, batch) in enumerate(iterator):
            src = batch.to(self.device)
            self.optim.zero_grad()
            output = self.net(src)
            center = center.to(self.device)
            loss = self.loss_mse(output, center)
            loss.backward()

            self.optim.step()
            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def _evaluate(self, iterator, center, epoch):

        self.net.eval()

        epoch_loss = 0

        lst_dist = []

        with torch.no_grad():
            for (i, batch) in enumerate(iterator):
                src = batch.to(self.device)
                output = self.net(src)
                if i == 0:
                    lst_emb = output
                else:
                    lst_emb = torch.cat((lst_emb, output), dim=0)
                center = center.to(self.device)
                loss = self.loss_mse(output, center)
                epoch_loss += loss.item()
                lst_dist.extend(get_dist(output, center))

        if epoch < 10:
            center = get_center(lst_emb)
            center[(abs(center) < self.eps) & (center < 0)] = -self.eps
            center[(abs(center) < self.eps) & (center > 0)] = self.eps

        return epoch_loss / len(iterator), center, lst_dist

    def train_DeepSVDD(self, train_x):
        x_train, x_eval = train_test_split(train_x, test_size=0.1, random_state=42)
        ds_train = PointDataset(torch.tensor(x_train).float())
        df_eval = PointDataset(torch.tensor(x_eval).float())
        train_iter = DataLoader(ds_train, self.batch_size, shuffle=True, worker_init_fn=np.random.seed(42))
        eval_iter = DataLoader(df_eval, self.batch_size, shuffle=True, worker_init_fn=np.random.seed(42))

        best_eval_loss = float('inf')

        for epoch in tqdm(range(self.max_epoch)):
            if epoch == 0:
                center = torch.Tensor([0.0 for _ in range(self.out_dim)])
            if epoch > 9:
                center = fixed_center
            train_loss = self._train(train_iter, center)
            eval_loss, center, lst_dist = self._evaluate(eval_iter, center, epoch)

            if epoch == 9:
                fixed_center = center

            if eval_loss < best_eval_loss and epoch >= 9:
                best_eval_loss = eval_loss
                torch.save(self.net.state_dict(), './saved_models/DeepSVDD.pt')
                self.c = fixed_center.cpu()
                pd.DataFrame(fixed_center.cpu().numpy()).to_csv('./saved_models/DeepSVDD_center.csv')

    def load_model(self):
        self.net.load_state_dict(torch.load('./saved_models/DeepSVDD.pt'))
        self.net.to(self.device)
        self.c = torch.Tensor(
            pd.read_csv('./saved_models/DeepSVDD_center.csv', index_col=0).iloc[:, 0])

    def get_R(self, train_x):
        ds = PointDataset(torch.tensor(train_x).float())
        train_iter = DataLoader(ds, self.batch_size, shuffle=False)
        _, _, lst_dist = self._evaluate(train_iter, self.c, 20)
        self.R = np.quantile(np.array(lst_dist), 1 - self.nu)

    def predict(self, test_x):
        self.load_model()
        self.net.eval()
        ds = PointDataset(torch.tensor(test_x).float())
        test_iter = DataLoader(ds, self.batch_size, shuffle=False)
        _, _, lst_dist = self._evaluate(test_iter, self.c, 20)
        print(self.R)
        print(np.mean(lst_dist))
        print(max(lst_dist))
        pred = [0 if i <= self.R else 1 for i in lst_dist]
        return lst_dist, pred