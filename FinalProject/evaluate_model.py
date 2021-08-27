import numpy as np


def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return float(correct) / y_true.shape[0]


def cm2pr_binary(cm):
    precision = cm[0,0]/np.sum(cm[:,0])
    recall = cm[0,0]/np.sum(cm[0])
    return (precision, recall)