import numpy as np


def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return float(correct) / y_true.shape[0]


# confusion matrix
def my_confusion_matrix(y_true, y_pred):
    N = np.unique(y_true).shape[0]  # number of classes
    cm = np.zeros((N, N))
    for n in range(y_true.shape[0]):
        cm[y_true[n], y_pred[n]] += 1
    return cm


# confusion matrix to precision + recall
def cm2pr_binary(y_true, y_pred):
    cm = my_confusion_matrix(y_true, y_pred)
    print("confusion_matrix: ")
    print(cm)
    precision = cm[0, 0] / np.sum(cm[:, 0])
    recall = cm[0, 0] / np.sum(cm[0])
    return (precision, recall)
