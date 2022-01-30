import torch
from torch import nn


class PUNNLoss(nn.Module):
    def __init__(self, pi):
        super(PUNNLoss, self).__init__()
        self.pi = pi

    def forward(self, g, t):
        positive, unlabeled = t == 1, t == 0
        n_p = max([1, torch.sum(positive)])
        n_u = max([1, torch.sum(unlabeled)])
        gp = torch.log(1 + torch.exp(-g))
        gu = torch.log(1 + torch.exp(g))
        lossp = self.pi * torch.sum(gp * positive) / n_p
        lossu = (
            torch.sum(gu * unlabeled) / n_u - self.pi * torch.sum(gu * positive) / n_p
        )
        if lossu < 0:
            loss = -lossu
        else:
            loss = lossp + lossu
        return loss