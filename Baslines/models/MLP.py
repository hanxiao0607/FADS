from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def MLP(train_x, train_y, test_x, test_y, seed=42):
    clf = MLPClassifier(hidden_layer_sizes=(256, 64), random_state=seed, max_iter=200).fit(train_x, train_y)
    y_pred = clf.predict(test_x)

    print('Anomaly Detection report:')
    print(classification_report(y_true=test_y, y_pred=y_pred, digits=5))

