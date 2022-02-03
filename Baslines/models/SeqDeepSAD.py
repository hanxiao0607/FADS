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

class LogDataset(Dataset):
    def __init__(self, seqs, lens, ys, semi):
        super().__init__()
        self.seqs = seqs
        self.lens = lens
        self.ys = ys
        self.semi = semi

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.seqs[idx], self.lens[idx], self.ys[idx], self.semi[idx]


def get_iter(x_and_len, y, semi, batch_size=1024, shuffle=False):
    X, lens = x_and_len
    dataset = LogDataset(X, lens, y, semi)
    if shuffle == True:
        iter = DataLoader(dataset, batch_size, shuffle=True, worker_init_fn=np.random.seed(42))
    else:
        iter = DataLoader(dataset, batch_size, worker_init_fn=np.random.seed(42))
    return iter


def get_normal(ten, lst_label):
    new_ten = []
    if lst_label == None:
        return torch.tensor([])
    for ind, val in enumerate(lst_label):
        if val == 0:
            new_ten.append(ten[ind])
    try:
        return torch.stack(new_ten)
    except:
        print(new_ten)
        print(lst_label)


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


class classification_lstm(nn.Module):
    def __init__(self, input_dim, emb_dim, out_dim):
        super(classification_lstm, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, out_dim, 1, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input, lens):
        embedded = self.dropout(self.embedding(input))
        output, _ = self.rnn(embedded)
        prediction = [torch.mean(output[i, :lens[i], :], dim=0) for i in range(len(lens))]
        prediction = torch.stack(prediction)
        return prediction


class DeepSAD(object):
    def __init__(self, input_dim, emb_dim, out_dim, batch_size=1024, eps=0.1, max_epoch=50, nu=0.05, device='cuda:1', alpha=1.0):
        super().__init__()

        self.nu = nu
        self.R = 0.0
        self.c = None
        self.eps = eps
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.device = device
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.alpha = alpha

        self.net = classification_lstm(self.input_dim, self.emb_dim, self.out_dim).to(self.device)

        self.optim = optim.Adam(self.net.parameters(), weight_decay=1e-6)
        self.loss_mse = nn.MSELoss()

    def _train(self, iterator, center):
        self.net.train()

        epoch_loss = 0

        for (i, batch) in enumerate(iterator):
            src = batch[0].to(self.device)
            lens = batch[1]
            label = batch[2].to(self.device)
            self.optim.zero_grad()
            output = self.net(src, lens)
            center = center.to(self.device)
            dist = torch.sum((output - center) ** 2, dim=1)
            loss = torch.where(label == 0, dist, dist ** (-1))
            loss = torch.mean(loss)
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
                src = batch[0].to(self.device)
                lens = batch[1]
                label = batch[2].to(self.device)
                output = self.net(src, lens)
                new_output = get_normal(output, label)
                if i == 0:
                    lst_emb = new_output
                else:
                    lst_emb = torch.cat((lst_emb, new_output), dim=0)
                center = center.to(self.device)
                dist = torch.sum((output - center) ** 2, dim=1)
                loss = torch.where(label == 0, dist, dist ** (-1))
                loss = torch.mean(loss)
                epoch_loss += loss.item()
                lst_dist.extend(get_dist(output, center))

        if epoch < 10:
            center = get_center(lst_emb)
            center[(abs(center) < self.eps) & (center < 0)] = -self.eps
            center[(abs(center) < self.eps) & (center > 0)] = self.eps

        return epoch_loss / len(iterator), center, lst_dist

    def train_DeepSAD(self, train_x, train_y, unlabel_x):
        train_semi = np.concatenate([[0 for _ in range(len(train_x))], [1 for _ in range(len(unlabel_x))]], axis=0)
        train_x = np.concatenate([train_x, unlabel_x], axis=0)
        train_y = np.concatenate([train_y, [0 for _ in range(len(unlabel_x))]], axis=0)
        x_train, x_eval, y_train, y_eval, semi_train, semi_eval = train_test_split(train_x, train_y, train_semi,
                                                                                   test_size=0.1, random_state=42)
        train_iter = get_iter(list_to_tensor(x_train), y_train, semi_train, self.batch_size, shuffle=False)
        eval_iter = get_iter(list_to_tensor(x_eval), y_eval, semi_eval, self.batch_size, shuffle=False)

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
                torch.save(self.net.state_dict(), './saved_models/DeepSAD.pt')
                self.c = fixed_center.cpu()
                pd.DataFrame(fixed_center.cpu().numpy()).to_csv('./saved_models/DeepSAD_center.csv')

    def load_model(self):
        self.net.load_state_dict(torch.load('./saved_models/DeepSAD.pt'))
        self.net.to(self.device)
        self.c = torch.Tensor(
            pd.read_csv('./saved_models/DeepSAD_center.csv', index_col=0).iloc[:, 0])

    def get_R(self, train_x, train_y):
        semi_train = [0 for _ in range(len(train_y))]
        train_iter = get_iter(list_to_tensor(train_x), train_y, semi_train, self.batch_size, shuffle=False)
        _, _, lst_dist = self._evaluate(train_iter, self.c, 20)
        self.R = np.quantile(np.array(lst_dist), 1 - self.nu)

    def predict(self, test_x):
        self.load_model()
        self.net.eval()
        semi_test = [0 for _ in range(len(test_x))]
        test_y = [0 for _ in range(len(test_x))]
        test_iter = get_iter(list_to_tensor(test_x), test_y, semi_test, self.batch_size)
        _, _, lst_dist = self._evaluate(test_iter, self.c, 20)
        print(self.R)
        print(np.mean(lst_dist))
        print(max(lst_dist))
        pred = [0 if i <= self.R else 1 for i in lst_dist]
        return lst_dist, pred