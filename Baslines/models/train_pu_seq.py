import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from Baslines.models.pu import PUNNLoss
import random
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from Baslines import utils



def train_epoch(x, len, y, model, loss_func, optimizer, use_gpu):
    optimizer.zero_grad()

    if use_gpu:
        x, len, y = x.cuda(), len, y.cuda()

    g = model(x, len)
    loss = loss_func(g, y.reshape(-1, 1))
    loss.backward()
    optimizer.step()

    return loss.item()


def train_pn(model, train_dataloader, test_dataloader, optimizer, epochs, use_gpu):
    loss_func = lambda g, y: torch.mean(torch.log(1 + torch.exp(-y * g)))

    if use_gpu:
        model.cuda()

    # Train
    model.train()

    pbar = tqdm(range(epochs), desc="Train PN")
    for e in pbar:
        for x, len, y in train_dataloader:
            loss = train_epoch(x, len, y, model, loss_func, optimizer, use_gpu)

        pbar.set_postfix({"loss": loss})
    pbar.close()

    # Eval
    model.eval()

    prob_list = []
    with torch.no_grad():
        for (x, len, y) in test_dataloader:
            if use_gpu:
                x = x.cuda()

            p = torch.sigmoid(model(x, len))

            prob_list.append(p.data.cpu().numpy())

    prob = np.concatenate(prob_list, axis=0)

    return model, prob


def test_pu(model, dataloader, quant, use_gpu, pi=0, epochs=0):
    theta = 0
    p_list, y_list = [], []

    with torch.no_grad():
        for x, len, y in dataloader:
            if use_gpu:
                x = x.cuda()
            p = model(x, len)
            p_list.append(p.detach().cpu().numpy())
            y_list.append(y.numpy())
        y = np.concatenate(y_list, axis=0)
        prob = np.concatenate(p_list, axis=0)
        if quant is True:
            temp = np.copy(prob).flatten()
            temp = np.sort(temp)
            theta = temp[np.int(np.floor(prob.shape[0] * (1 - pi)))]


    pred = np.zeros(prob.shape[0])
    pred[(prob > theta).flatten()] = 1
    accuracy = np.mean(pred == y)
    precision = np.sum((pred == y)[pred == 1]) / np.sum(pred == 1)
    recall = np.sum((pred == y)[y == 1]) / np.sum(y == 1)
    if epochs == 19:
        print('Anomaly Detection report:')
        print(classification_report(y_true=y, y_pred=pred, digits=5))
        print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(y, pred)))
        print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(y, pred)))
        print('Anomaly Detection FPR-AT-95-TPR: {:.5f}'.format(utils.getfpr95tpr(y_true=y, dist=prob.reshape(-1))))

    return accuracy, precision, recall


def train_pu(pi, model, train_dataloader, test_dataloader, optimizer, epochs, use_gpu):
    loss_list = []
    acc_list = []
    acc_quant_list = []
    pre_list = []
    rec_list = []
    pre_quant_list = []
    rec_quant_list = []

    loss_func = PUNNLoss(pi)

    if use_gpu:
        model.cuda()
    pbar = tqdm(range(epochs), desc="Train PU")
    for e in pbar:
        loss_step = 0
        count = 0

        # Train
        model.train()
        for x, len, y in train_dataloader:
            loss = train_epoch(x, len, y, model, loss_func, optimizer, use_gpu)
            loss_step += loss
            count += 1

        loss_step /= count
        loss_list.append(loss_step)

        # Eval
        model.eval()
        acc, pre, rec = test_pu(model, test_dataloader, False, use_gpu, epochs=e)
        acc_quant, pre_quant, rec_quant = test_pu(
            model, test_dataloader, True, use_gpu, pi
        )

        acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)

        acc_quant_list.append(acc_quant)
        pre_quant_list.append(pre_quant)
        rec_quant_list.append(rec_quant)

        pbar.set_postfix({"loss": loss_step, "acc": acc, "acc_quant": acc_quant})

    pbar.close()

    loss_list = np.array(loss_list)

    acc_list = np.array(acc_list)
    pre_list = np.array(pre_list)
    rec_list = np.array(rec_list)

    acc_quant_list = np.array(acc_quant_list)
    pre_quant_list = np.array(pre_quant_list)
    rec_quant_list = np.array(rec_quant_list)

    return (
        model,
        loss_list,
        acc_list,
        pre_list,
        rec_list,
        acc_quant_list,
        pre_quant_list,
        rec_quant_list,
    )


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


def get_iter(x_and_len, y, batch_size=1024, shuffle=False):
    X, lens = x_and_len
    dataset = LogDataset(X, lens, y)
    if shuffle == True:
        iter = DataLoader(dataset, batch_size, shuffle=True, worker_init_fn=np.random.seed(42))
    else:
        iter = DataLoader(dataset, batch_size, worker_init_fn=np.random.seed(42))
    return iter


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


def train_model(
    pn_model,
    pu_model,
    pn_optimizer,
    pu_optimizer,
    x_train,
    y_train,
    x_test,
    y_test,
    pdata,
    epochs=100,
    batch_size=64,
    use_gpu=True,
):
    # Data Process
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    # pi = np.mean(y_train)
    pi = 0.01
    x = x_train.copy()
    x.extend(x_test)
    y = np.concatenate([y_train, y_test], axis=0).astype(int)
    x_p = []
    for i in range(len(x_train)):
        if y_train[i] == 1:
            x_p.append(x_train[i])

    # PN classification
    pn_train_dataset = get_iter(list_to_tensor(x), y, batch_size=batch_size, shuffle=True)
    pn_test_dataset = get_iter(list_to_tensor(x_p), [1 for _ in range(len(x_p))], batch_size=batch_size)
    pn_train_dataloader = pn_train_dataset
    pn_test_dataloader = pn_test_dataset
    pn_model, prob = train_pn(
        pn_model, pn_train_dataloader, pn_test_dataloader, pn_optimizer, epochs, use_gpu
    )

    prob /= np.mean(prob)
    prob /= np.max(prob)

    x_p_new = []
    rand = np.random.uniform(size=prob.shape)
    for i in range(len(x_p)):
        if prob[i] > rand[i]:
            x_p_new.append(x_p[i])
    x_p = x_p_new
    random.shuffle(x_p)
    x_p = x_p[:pdata]

    y_p = np.ones(len(x_p))
    y_u = np.zeros(len(x_train))

    x = x_train.copy()
    x.extend(x_p)
    y_train = np.concatenate([y_p, y_u], axis=0)

    pu_train_dataset = get_iter(list_to_tensor(x), y_train, batch_size=batch_size, shuffle=True)
    pu_test_dataset = get_iter(list_to_tensor(x_test), y_test, batch_size=batch_size, shuffle=True)

    pu_train_dataloader = pu_train_dataset
    pu_test_dataloader = pu_test_dataset

    (pu_model, loss, acc, pre, rec, acc_quant, pre_quant, rec_quant,) = train_pu(
        pi,
        pu_model,
        pu_train_dataloader,
        pu_test_dataloader,
        pu_optimizer,
        epochs,
        use_gpu,
    )

    return pn_model, pu_model, loss, acc, pre, rec, acc_quant, pre_quant, rec_quant