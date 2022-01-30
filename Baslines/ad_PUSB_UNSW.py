import utils
import torch
import numpy as np
from torch import nn, optim
from models.train_pu import train_model


def main():
    iters = 10  # Run multiple experiments to get average results
    pdata = 1000
    epochs = 200
    batch_size = 1024

    seed = 0

    use_gpu = torch.cuda.is_available()

    loss_pu = np.zeros((iters, epochs))
    est_error_pu = np.zeros((iters, epochs))
    est_error_pubp = np.zeros((iters, epochs))
    est_precision_pu = np.zeros((iters, epochs))
    est_recall_pu = np.zeros((iters, epochs))
    est_precision_pubp = np.zeros((iters, epochs))
    est_recall_pubp = np.zeros((iters, epochs))

    dir_UNSW = '../UNSW/Datasets/NUSW_small.csv'
    x_train, y_train, x_test, y_test = utils.preprocessing_UNSW_DeepSAD(dir_UNSW, n_sup=10, seed=seed, adc=0, pul=1)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)


    for i in range(iters):
        print("Experiment:", i)
        np.random.seed(seed)
        pn_model = nn.Sequential(
            nn.Linear(x_train.shape[1], 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

        pn_optimizer = optim.Adam(pn_model.parameters(), lr=1e-5)

        pu_model = nn.Sequential(
            nn.Linear(x_train.shape[1], 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

        pu_optimizer = optim.Adam(pu_model.parameters(), lr=1e-5, weight_decay=0.005)
        (
            pn_model,
            pu_model,
            loss,
            acc,
            pre,
            rec,
            acc_quant,
            pre_quant,
            rec_quant,
        ) = train_model(
            pn_model,
            pu_model,
            pn_optimizer,
            pu_optimizer,
            x_train,
            y_train,
            x_test,
            y_test,
            pdata,
            epochs,
            batch_size,
            use_gpu,
        )

        loss_pu[i] = loss
        est_error_pu[i] = acc
        est_precision_pu[i] = pre
        est_recall_pu[i] = rec

        est_error_pubp[i] = acc_quant
        est_precision_pubp[i] = pre_quant
        est_recall_pubp[i] = rec_quant

        seed += 1

    loss_pu_mean = np.mean(loss_pu, axis=0)
    est_error_pu_mean = np.mean(est_error_pu, axis=0)
    est_error_pubp_mean = np.mean(est_error_pubp, axis=0)
    est_error_pu_std = np.std(est_error_pu, axis=0)
    est_error_pubp_std = np.std(est_error_pubp, axis=0)

    return (
        loss_pu_mean,
        est_error_pu_mean,
        est_error_pubp_mean,
        est_error_pu_std,
        est_error_pubp_std,
    )


if __name__ == "__main__":
    (
        loss_pu_mean,
        est_error_pu_mean,
        est_error_pubp_mean,
        est_error_pu_std,
        est_error_pubp_std,
    ) = main()

    print("loss_pu_mean:")
    print(loss_pu_mean)
    print("est_error_pu_mean:")
    print(est_error_pu_mean)
    print("est_error_pubp_mean:")
    print(est_error_pubp_mean)