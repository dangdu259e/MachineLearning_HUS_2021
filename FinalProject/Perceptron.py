import numpy as np
import read_data

# tính đầu ra khi biết đầu vào x và weights w
def h(w, x):
    return np.sign(np.dot(w.T, x))


# kiểm tra xem thuật toán đã hội tụ chưa. Ta chỉ cần so sánh h(w, X) với ground truth y. Nếu giống nhau thì dừng thuật toán.
def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)


# hàm chính thực hiện PLA.
def perceptron(X, y):
    # kích thước của ma trận đầu vào
    N = X.shape[1]
    d = X.shape[0]

    # Chọn ngẫu nhiên một vector hệ số  w với các phần tử gần 0
    w_init = np.random.randn(d, 1)

    # thêm vector ngẫu nhiên vừa khởi tạo w_init vào trong mảng các w
    w = [w_init]

    # điểm dữ liệu đầu vào x bị phân loại sai lớp
    mis_points = []
    while True:
        # mix data
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi)[0] != yi:  # misclassified point
                mis_points.append(mix_id[i])
                # thực hiện cập nhật bộ hệ số w theo công thức w = w + eta*(yi*xi)
                # trong đó tốc độ học eta => được cố định  = 1 theo mô hình
                w_new = w[-1] + yi * xi
                w.append(w_new)

        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)


# d = X.shape[0]
# w_init = np.random.randn(d, 1)
# (w, m) = perceptron(X, y, w_init)
if __name__ == '__main__':
    url_data = "./data/data.txt"
    data, X_train, y_train, X_test, y_test = read_data.read_data(url_data)
    d = X_train.shape[0]

    (w, m) = perceptron(X_train, y_train)