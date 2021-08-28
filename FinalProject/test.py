"""
  Homeworkweek3: Machine Learning 2021
  26/07/2021

  __author__  = "Đặng Trung Du"
  __MSV__     = "18001108"
  __Speciality__= "Computer and Information Science"
  __Class__   = "K63A3_MIM_HUS"
"""

# import lib
import numpy as np
# import dataset
import pandas as pd
import math

df = pd.read_csv(
    'http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data', 
    sep=',', 
    header=None
)
# format data input
X = df.iloc[:, :57].to_numpy()
y = df[57].to_numpy()
y.reshape((4601,1))
y = np.array([y]).T

# Hàm sigmoid
def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def accuracy(theta, x_train, y_train):
  c = 0
  n = len(x_train)
  for i in range(n):
    xi_vec = np.array([x_train[i]]).T
    yi = y_train[i][0]
    htheta_x = sigmoid(np.dot(theta.T, xi_vec))
    if (htheta_x >= 0.5 and yi == 1):
      c += 1
    if (htheta_x < 0.5 and yi == 0):
      c += 1
  return str(100*c/n) + "%"


# Stochastic Gradient Descent
def logistic_SGD_regression(X, y, alpha):
    [N, d] = X.shape
    theta = [np.zeros([d, 1], dtype=float)]
    # đặt maxrepeat = 1000 (chọn số ngẫu nhiên có thể là 10000 ,...)
    for repeat in range(100):
        k = np.random.permutation(N)
        for i in k:
            theta_vector = theta[-1]
            xi_vec = np.array([X[i]]).T
            htheta_x = sigmoid(np.dot(theta_vector.T, xi_vec))
            htheta_x_yi = htheta_x - y[i][0]
            theta_new = theta_vector - alpha * htheta_x_yi * xi_vec
            theta.append(theta_new)
    return theta[-1]


X_test = np.array(X).T
X_test = np.concatenate((np.ones((1, X_test.shape[1])), X_test), axis=0)
X_test = X_test.T
print(X_test.shape)
y_test = y
print(y_test.shape)
alpha = 10**-6
theta = logistic_SGD_regression(X_test, y_test, alpha)
print(theta.T)
print("accuracy: "+ str(accuracy(theta, X_test, y_test)))