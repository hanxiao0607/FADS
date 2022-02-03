from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def SVM(train_x, train_y, test_x, test_y, seed=42):
    clf = SVC(probability=True, decision_function_shape='ovr', random_state=seed)
    clf.fit(train_x, train_y)
    score = clf.predict_proba(test_x)
    y_pred = clf.predict(test_x)

    print('Anomaly Detection report:')
    print(classification_report(y_true=test_y, y_pred=y_pred, digits=5))
    # print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(test_y, score, multi_class="ovr")))
    # print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(test_y, score)))