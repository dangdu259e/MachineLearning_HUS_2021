# import lib
import numpy as np
import read_data
import evaluate_model


# Hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Stochastic Gradient Descent
def logistic_SGD_regression(X, y, alpha):
    # alpha: learning rate => tốc độ học
    # thêm cột 1 vào ma trận đầu vào X
    X = np.array(X).T
    X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
    X = X.T

    # chuyển ma trận y thành vector cột | tương ứng với từng hàng của vector X
    y = np.array([y]).T

    # kích thước của ma trận đầu vào
    [N, d] = X.shape
    # khởi tạo theta ngẫu nhiên ~ 0
    theta = [np.zeros([d, 1], dtype=float)]
    # đặt maxrepeat = 1000 (chọn số ngẫu nhiên có thể là 10000 ,...)
    for repeat in range(5000):
        # Xáo trộn dữ liệu đảm bảo tính ngẫu nhiên
        k = np.random.permutation(N)
        for i in k:
            theta_vector = theta[-1]
            xi_vec = np.array([X[i]]).T
            htheta_x = sigmoid(np.dot(theta_vector.T, xi_vec))
            htheta_x_yi = htheta_x - y[i][0]
            theta_new = theta_vector - alpha * htheta_x_yi * xi_vec
            theta.append(theta_new)
    return theta[-1].T


# hàm chuyển đầu ra của class y thành 0 và 1
def formart_class(y):
    y_new = []
    for i in y:
        if (i == 2):
            y_new.append(0)
        else:
            y_new.append(1)
    result = np.array(y_new)
    return result


# dự đoán đầu ra theo x
# sigmoid(w^T * X_test)
def guess_output(w, X_test):
    # thêm cột 1 vào ma trận đầu vào X
    X_test = np.array(X_test).T
    X_test = np.concatenate((np.ones((1, X_test.shape[1])), X_test), axis=0)
    X_test = X_test.T
    # công thức bị thay đổi một chút vì đầu ra mình đã chuyển theta thành hàng ngang
    # xs rơi vào lớp 1 (cách tính  kq< 0.5 => lớp 0, kq >= 0.5 => lớp 1)
    y_pred = sigmoid(np.dot(w, X_test.T))
    result = []
    for i in y_pred[0]:
        if (i < 0.5):
            result.append(0)
        else:
            result.append(1)
    return np.array(result)


if __name__ == '__main__':
    url_data = "./data/data.txt"
    data, X_train, y_train, X_test, y_test = read_data.read_data(url_data)

    # formart result => chuyển thành tập đích là 0 và 1
    y_train = formart_class(y_train)
    y_test = formart_class(y_test)

    # SGD => tối ưu hàm mất mát
    print("Logistic Stochastic Gradient Descent Regression")
    w = logistic_SGD_regression(X_train, y_train, 10 ** -6)
    y_pred = guess_output(w, X_test)
    print("y_pred: ")
    print(y_pred)
    print("y_origin: ")
    print(y_test)
    accuracy = evaluate_model.accuracy(y_true=y_test, y_pred=y_pred)
    print("accuracy = " + str(accuracy * 100) + "%")
    (precision, recall) = evaluate_model.cm2pr_binary(y_true=y_test, y_pred=y_pred)
    print("precision = " + str(precision))
    print("recall = " + str(recall))
