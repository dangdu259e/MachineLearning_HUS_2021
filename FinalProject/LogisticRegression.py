import numpy as np
import read_data
import math
np.random.seed(2)


# hàm activation sìgmoid
def sigmoid(s):
    return 1 / (1 + math.exp(-s))


def logistic_SGD_regression(X, y, alpha):
    term = X.T
    term = np.concatenate((np.ones((1, term.shape[1])), term), axis=0)
    # thực hiện thêm cột 1 vào đầu ma trận X
    X = term.T

    # kích thước của ma trận đầu vào
    [N, d] = X.shape

    # khởi tạo vector theta đầu tiên ~ 0
    theta = [np.zeros([d, 1], dtype=float)]
    print(theta[0].shape)
    # đặt maxrepeat = 1000 (chọn số ngẫu nhiên có thể là 10000 ,...)
    for repeat in range(1000):

        # xáo trộn tập dữ liệu đầu vào đảm bảo tính ngẫu nhiên
        k = np.random.permutation(N)
        for i in k:
            theta_vector = theta[-1]
            xi_vec = np.array([X[i]]).T
            htheta_x = sigmoid(np.dot(theta_vector.T, xi_vec))
            htheta_x_yi = htheta_x - y[i][0]

            # theta được cập nhật theo công thức
            theta_new = theta_vector - alpha * htheta_x_yi * xi_vec
            theta.append(theta_new)
    return theta[-1]


# kiểm tra mô hình khi đã trainning mô hình xong
def logistic_reg_w(w, X_test, y_test):
    y_pred = sigmoid(np.dot(w.T, X_test))
    print("dự đoán: ")
    print(y_pred)
    print("origin: ")
    print(y_test)


# kiểm tra mô hình với bộ dữ liệu test và bộ trainning data
def logistic_reg(X_train, y_train, X_test, y_test, alpha):
    w = logistic_SGD_regression(X_train, y_train, alpha)
    y_pred = sigmoid(np.dot(w.T, X_test))
    print("dự đoán: ")
    print(y_pred)
    print("origin: ")
    print(y_test)


def formart_class(y):
    y_new = []
    for i in y:
        if (i == 2):
            y_new.append(0)
        else:
            y_new.append(1)
    y_new = np.array(y_new)
    return y_new


if __name__ == '__main__':
    url_data = "./data/data.txt"
    data, X_train, y_train, X_test, y_test = read_data.read_data(url_data)

    # y_train đang có dạng là 2 hoặc 4 -> cần phải format y thành các giá trị 0 1 tương ứng
    y_train = formart_class(y_train)
    y_test = formart_class(y_test)
    print(X_train.shape)
    print(y_train.shape)

    alpha = 10**-6  # hệ số học (learning rate)
    w = logistic_SGD_regression(X_train, y_train, alpha)
    print(w.shape)
    logistic_reg_w(w, X_test, y_test)
