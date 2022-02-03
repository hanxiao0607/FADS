from models import utils, DNN, SeqDNN
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

def main():
    dir_UNSW = '../UNSW/Datasets/NUSW_small.csv'
    lst = list(range(10))
    print('UNSW')
    for i in lst:
        seen_x, seen_y, test_x, test_y = utils.preprocessing_UNSW(dir_UNSW, n_sup=10, seed=i, adc=1, unsupervised=0, oversampling=1)
        mlp = DNN.MLP(input_dim=293, output_dim=6, max_epoch=20, batch_size=1024)
        mlp.train_MLP(seen_x, seen_y)
        y_pred = mlp.predict(test_x, test_y)
        print('Anomaly Detection report:')
        print(classification_report(y_true=test_y, y_pred=y_pred, digits=5))
    dir_IDS = '../IDS/Datasets/IDS2018_small.csv'
    print('-'*20)
    print('IDS')
    for i in lst:
        seen_x, seen_y, test_x, test_y = utils.preprocessing_IDS(dir_IDS, n_sup=10, seed=i, adc=1, unsupervised=0, oversampling=1)
        mlp = DNN.MLP(input_dim=77, output_dim=4, max_epoch=20, batch_size=1024)
        mlp.train_MLP(seen_x, seen_y)
        y_pred = mlp.predict(test_x, test_y)
        print('Anomaly Detection report:')
        print(classification_report(y_true=test_y, y_pred=y_pred, digits=5))
    dir_CERT = '../CERT_EMB/Datasets/final_data.csv'
    print('-'*20)
    print('CERT')
    for i in lst:
        seen_x, seen_y, _, test_x, test_y = utils.preprocessing_CERT_EMB_DeepSAD(dir_CERT, n_sup=10, seed=i, adc=1, unsupervised=0)
        mlp = SeqDNN.MLP(input_dim=24, emb_dim=64, hid_dim=128, output_dim=5, max_epoch=20, batch_size=128)
        mlp.train_MLP(seen_x, seen_y)
        y_pred = mlp.predict(test_x, test_y)
        print('Anomaly Detection report:')
        print(classification_report(y_true=test_y, y_pred=y_pred, digits=5))
    print('done')


if __name__ == '__main__':
    main()
