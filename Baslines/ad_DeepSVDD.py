from models import utils, DeepSVDD, SeqDeepSVDD
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from Baslines import utils


def main():
    # dir_UNSW = '../UNSW/Datasets/NUSW_small.csv'
    lst = list(range(10))
    # print('UNSW')
    # for i in lst:
    #     seen_x, seen_y, test_x, test_y = utils.preprocessing_UNSW(dir_UNSW, n_sup=10, seed=i, adc=0)
    #     deepsvdd = DeepSVDD.DeepSVDD(input_dim=293, out_dim=128, max_epoch=50, batch_size=1024, nu=0.05)
    #     deepsvdd.train_DeepSVDD(seen_x)
    #     deepsvdd.get_R(seen_x)
    #     y_dist, y_pred = deepsvdd.predict(test_x)
    #     print('Anomaly Detection report:')
    #     print(classification_report(y_true=test_y, y_pred=y_pred, digits=5))
    #     print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(test_y, y_pred)))
    #     print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(test_y, y_pred)))
    #     print('Anomaly Detection FPR-AT-95-TPR: {:.5f}'.format(
    #         utils.getfpr95tpr(y_true=test_y, dist=y_dist)))
    dir_IDS = '../IDS/Datasets/IDS2018_small.csv'
    print('-'*20)
    print('IDS')
    for i in lst:
        seen_x, seen_y, test_x, test_y = utils.preprocessing_IDS(dir_IDS, n_sup=10, seed=i, adc=0)
        deepsvdd = DeepSVDD.DeepSVDD(input_dim=77, out_dim=128, batch_size=1024, eps=0.1, max_epoch=50, nu=0.10)
        deepsvdd.train_DeepSVDD(seen_x)
        deepsvdd.get_R(seen_x)
        y_dist, y_pred = deepsvdd.predict(test_x)
        print('Anomaly Detection report:')
        print(classification_report(y_true=test_y, y_pred=y_pred, digits=5))
        print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(test_y, y_pred)))
        print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(test_y, y_pred)))
        print('Anomaly Detection FPR-AT-95-TPR: {:.5f}'.format(
            utils.getfpr95tpr(y_true=test_y, dist=y_dist)))
    dir_CERT = '../CERT/Datasets/CERT52_small.csv'
    # print('-'*20)
    # print('CERT')
    # for i in lst:
    #     seen_x, seen_y, test_x, test_y = utils.preprocessing_CERT_EMB_DeepSVDD(dir_CERT, n_sup=10, seed=i, adc=0, unsupervised=1)
    #     deepsvdd = SeqDeepSVDD.DeepSVDD(input_dim=24, emb_dim=64, out_dim=128, max_epoch=200, batch_size=128, nu=0.99)
    #     deepsvdd.train_DeepSVDD(seen_x)
    #     deepsvdd.get_R(seen_x)
    #     y_dist, y_pred = deepsvdd.predict(test_x)
    #     print('Anomaly Detection report:')
    #     print(classification_report(y_true=test_y, y_pred=y_pred, digits=5))
    #     print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(test_y, y_pred)))
    #     print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(test_y, y_pred)))
    #     print('Anomaly Detection FPR-AT-95-TPR: {:.5f}'.format(
    #         utils.getfpr95tpr(y_true=test_y, dist=y_dist)))
    print('done')


if __name__ == '__main__':
    main()