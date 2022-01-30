import utils
from pulearn import ElkanotoPuClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def main():
    dir_UNSW = '../UNSW/Datasets/NUSW_small.csv'
    lst = list(range(10))
    print('UNSW')
    for i in lst:
        X, y, test_x, test_y = utils.preprocessing_UNSW_DeepSAD(dir_UNSW, n_sup=10, seed=i, adc=0, pul=1)
        svc = SVC(probability=True)
        pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
        pu_estimator.fit(X, y)
        scaler = MinMaxScaler()
        y_pred = pu_estimator.predict(test_x)
        y_pred = np.where(y_pred < 0, 0, y_pred)
        y_prob = scaler.fit_transform(pu_estimator.predict_proba(test_x).reshape(-1, 1))
        print('Anomaly Detection report:')
        print(classification_report(y_true=test_y, y_pred=y_pred, digits=5))
        print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(test_y, y_pred)))
        print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(test_y, y_pred)))
    dir_IDS = '../IDS/Datasets/IDS2018_small.csv'
    print('-'*20)
    print('IDS')
    for i in lst:
        X, y, test_x, test_y = utils.preprocessing_IDS_DeepSAD(dir_IDS, n_sup=10, seed=i, adc=0, pul=1)
        svc = SVC(probability=True)
        pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
        pu_estimator.fit(X, y)
        scaler = MinMaxScaler()
        y_pred = pu_estimator.predict(test_x)
        y_pred = np.where(y_pred < 0, 0, y_pred)
        y_prob = scaler.fit_transform(pu_estimator.predict_proba(test_x).reshape(-1, 1))
        print('Anomaly Detection report:')
        print(classification_report(y_true=test_y, y_pred=y_pred, digits=5))
        print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(test_y, y_pred)))
        print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(test_y, y_pred)))
    dir_CERT = '../CERT/Datasets/CERT52_small.csv'
    print('-'*20)
    print('CERT')
    for i in lst:
        X, y, test_x, test_y = utils.preprocessing_CERT_EMB_PUL(dir_CERT, n_sup=10, seed=i)
        svc = SVC(probability=True)
        pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
        pu_estimator.fit(X, y)
        scaler = MinMaxScaler()
        y_pred = pu_estimator.predict(test_x)
        y_pred = np.where(y_pred < 0, 0, y_pred)
        y_prob = scaler.fit_transform(pu_estimator.predict_proba(test_x).reshape(-1, 1))
        print('Anomaly Detection report:')
        print(classification_report(y_true=test_y, y_pred=y_pred, digits=5))
        print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(test_y, y_pred)))
        print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(test_y, y_pred)))
    print('done')


if __name__ == '__main__':
    main()
