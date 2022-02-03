from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import numpy as np
from Baslines import utils

def SVM(train_x, train_y, test_x, test_y):
    clf = OneClassSVM()
    clf.fit(train_x)
    score = clf.score_samples(test_x)
    y_pred = clf.predict(test_x)

    y_pred = np.where(y_pred == 1, 0, 1)

    print('Anomaly Detection report:')
    print(classification_report(y_true=test_y, y_pred=y_pred, digits=5))
    print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(test_y, y_pred)))
    print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(test_y, y_pred)))
    print('Anomaly Detection FPR-AT-95-TPR: {:.5f}'.format(
        utils.getfpr95tpr(y_true=test_y, dist=score)))