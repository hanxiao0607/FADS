import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader
from Baslines.models.pu import PUNNLoss
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from Baslines import utils



def train_epoch(x, y, model, loss_func, optimizer, use_gpu):
    optimizer.zero_grad()

    if use_gpu:
        x, y = x.cuda(), y.cuda()

    g = model(x)
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
        for x, y in train_dataloader:
            loss = train_epoch(x, y, model, loss_func, optimizer, use_gpu)

        pbar.set_postfix({"loss": loss})
    pbar.close()

    # Eval
    model.eval()

    prob_list = []
    with torch.no_grad():
        for (x,) in test_dataloader:
            if use_gpu:
                x = x.cuda()

            p = torch.sigmoid(model(x))

            prob_list.append(p.data.cpu().numpy())

    prob = np.concatenate(prob_list, axis=0)

    return model, prob


def test_pu(model, dataloader, quant, use_gpu, pi=0, epochs=0):
    theta = 0
    p_list, y_list = [], []

    with torch.no_grad():
        for x, y in dataloader:
            if use_gpu:
                x = x.cuda()
            p = model(x)
            p_list.append(p.data.cpu().numpy())
            y_list.append(y.numpy())

        y = np.concatenate(y_list, axis=0)
        prob = np.concatenate(p_list, axis=0)
        if quant is True:
            temp = np.copy(prob).flatten()
            temp = np.sort(temp)
            theta = temp[np.int(np.floor(len(prob) * (1 - pi)))]
    pred = np.zeros(len(prob))
    pred[(prob > theta).flatten()] = 1
    accuracy = np.mean(pred == y)
    precision = np.sum((pred == y)[pred == 1]) / np.sum(pred == 1)
    recall = np.sum((pred == y)[y == 1]) / np.sum(y == 1)
    if epochs == 199:
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
        for x, y in train_dataloader:
            loss = train_epoch(x, y, model, loss_func, optimizer, use_gpu)
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

    pi = np.mean(y_train)

    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    x_p = x_train[y_train == 1]

    # PN classification
    pn_train_dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
    pn_test_dataset = TensorDataset(torch.tensor(x_p))
    pn_train_dataloader = DataLoader(
        pn_train_dataset, batch_size=batch_size, shuffle=True
    )
    pn_test_dataloader = DataLoader(pn_test_dataset, batch_size=batch_size)
    pn_model, prob = train_pn(
        pn_model, pn_train_dataloader, pn_test_dataloader, pn_optimizer, epochs, use_gpu
    )

    prob /= np.mean(prob)
    prob /= np.max(prob)

    rand = np.random.uniform(size=prob.shape)
    x_p = x_p[(prob > rand).flatten()]
    perm = np.random.permutation(len(x_p))
    x_p = x_p[perm[:pdata]]

    y_p = np.ones(len(x_p))
    y_u = np.zeros(len(x_train))

    x_train = np.concatenate([x_p, x_train], axis=0)
    y_train = np.concatenate([y_p, y_u], axis=0)

    pu_train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    pu_test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

    pu_train_dataloader = DataLoader(
        pu_train_dataset, batch_size=batch_size, shuffle=True
    )
    pu_test_dataloader = DataLoader(
        pu_test_dataset, batch_size=batch_size, shuffle=True
    )

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