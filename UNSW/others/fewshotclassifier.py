import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


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
        return self.seqs[idx], self.ys[idx]


def get_iter(X, y, batch_size=1024, shuffle=False):
    if y is None:
        y = [-1 for _ in range(len(X))]
    dataset = LogDataset(X, y)
    if shuffle == True:
        iter = DataLoader(dataset, batch_size, shuffle=True, worker_init_fn=np.random.seed(42))
    else:
        iter = DataLoader(dataset, batch_size)
    return iter


def list_to_tensor(lst):
    return torch.tensor(np.array([np.array(i) for i in lst])).float()


def extract_sample(n_way, n_support, n_query, datax, datay):
    """
    Picks random sample of size n_support+n_querry, for n_way classes
    Args:
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        datax (np.array): dataset of seqs
        datay (np.array): dataset of labels
    Returns:
        (dict) of:
            (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
            (int): n_way
            (int): n_support
            (int): n_query
    """
    sample = []
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    for cls in K:
        datax_cls = datax[datay == cls]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support + n_query)]
        sample.append([np.array(i) for i in sample_cls])
    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()
    return ({
        'seqs': sample,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    })


class FewShotClassifierNet(nn.Module):
    def __init__(self, options):
        super(FewShotClassifierNet, self).__init__()
        self.fc1 = nn.Linear(options['input_dim'], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, options['n_ways'])
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class FewShotClassifier(nn.Module):
    def __init__(self, options):
        super(FewShotClassifier, self).__init__()
        self.input_dim = options['input_dim']
        self.n_ways = options['n_ways']
        self.n_support = options['n_support']
        self.n_query = options['n_query']
        self.device = options['device']
        self.net = FewShotClassifierNet(options)
        self.optim = optim.Adam(self.net.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.max_epoch = options['max_epoch']
        self.epoch_size = options['epoch_size']

    def fsc_train(self, train_x, train_y, path='fsc.pt'):
        self.net.train()
        self.net.to(self.device)
        epoch = 0
        stop = False

        while epoch < self.max_epoch and not stop:
            running_loss = 0.0
            min_loss = np.inf

            for episode in range(self.epoch_size):
                sample = extract_sample(self.n_ways, self.n_support, self.n_query, train_x, train_y)
                self.optim.zero_grad()
                sample_seqs = sample['seqs'].to(self.device)
                x_support = sample_seqs[:, :self.n_support]
                x_query = sample_seqs[:, self.n_support:]
                target_inds = torch.arange(0, self.n_ways).view(self.n_ways, 1, 1).expand(self.n_ways, self.n_query,
                                                                                        1).long()
                target_inds = Variable(target_inds, requires_grad=False)
                target_inds = target_inds.to(self.device)

                x = torch.cat([x_support.contiguous().view(self.n_ways * self.n_support, *x_support.size()[2:]),
                               x_query.contiguous().view(self.n_ways * self.n_query, *x_query.size()[2:])], 0)

                output = self.net.forward(x)

                output_query = output[self.n_ways * self.n_support:]

                loss = self.criterion(output_query, target_inds.squeeze().view(-1))
                loss.backward()
                self.optim.step()
                running_loss += loss.item()
            epoch_loss = running_loss / self.epoch_size
            print(f'epoch loss {epoch_loss}')
            if min_loss >= epoch_loss:
                min_loss = epoch_loss
                torch.save(self.net, './saved_models/' + path)
            epoch += 1

    def fsc_test(self, test_x, test_y, path='fsc.pt'):
        self.net = torch.load('./saved_models/' + path)
        self.net.to(self.device)
        self.net.eval()
        test_iter = get_iter(list_to_tensor(test_x), test_y)
        y_pred = []
        y_true = []
        y_vals = []
        for batch in test_iter:
            src = batch[0].to(self.device)
            y_true.extend(list(batch[1]))
            output = self.net.forward(src)
            val, pred = torch.max(output, 1)
            y_vals.extend(val.detach().cpu().numpy())
            y_pred.extend(pred.detach().cpu().numpy())
        return y_true, y_pred, y_vals
