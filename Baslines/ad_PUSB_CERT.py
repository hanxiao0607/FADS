import utils
import torch
import numpy as np
from torch import nn, optim
from models.train_pu_seq import train_model

class classification_lstm(nn.Module):
    def __init__(self):
        super(classification_lstm, self).__init__()
        self.embedding = nn.Embedding(24, 64, padding_idx=0)
        self.rnn = nn.GRU(64, 128, 1, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)

    def forward(self, input, lens):
        embedded = self.dropout(self.embedding(input))
        output, _ = self.rnn(embedded)
        prediction = [torch.mean(output[i, :lens[i], :], dim=0) for i in range(len(lens))]
        prediction = self.fc(torch.stack(prediction))
        return prediction

def main():
    iters = 10  # Run multiple experiments to get average results
    pdata = 1000
    epochs = 20
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

    dir_CERT = '../CERT/Datasets/CERT52_small.csv'
    x_train, y_train, x_test, y_test = utils.preprocessing_CERT_EMB_PUL(dir_CERT, n_sup=10, seed=seed)
    for i in range(iters):
        print("Experiment:", i)
        np.random.seed(seed)
        pn_model = classification_lstm()

        pn_optimizer = optim.Adam(pn_model.parameters(), lr=1e-5)

        pu_model = classification_lstm()

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