import numpy as np
import read_data
import evaluate_model


# tính đầu ra khi biết đầu vào x và weights w
def h(w, X):
    X = X.T
    N = X.shape[1]

    # creat x0 = (1,1,1,1,1,1,1,1,1,.....,1)
    x0 = np.ones((1, N))

    # thêm hàng x0 = 1 vào
    X = np.concatenate((x0, X))

    return np.sign(np.dot(w.T, X))


# kiểm tra xem thuật toán đã hội tụ chưa. Ta chỉ cần so sánh h(w, X) với ground truth y.
# Nếu giống nhau thì dừng thuật toán.
def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)


# hàm chính thực hiện PLA.
def perceptron(X, y):
    # chuyển vị ma trận X
    # các phần tử trong x là các vector cột
    X = X.T

    # Chuyển y thành mảng 2 chiều thôi tại y đang là mảng 1 chiều
    term = []
    term.append(y)
    y = np.array(term)
    # N là số điểm dữ liệu
    N = X.shape[1]

    # creat x0 = (1,1,1,1,1,1,1,1,1,.....,1)
    x0 = np.ones((1, N))

    # thêm hàng x0 = 1 vào
    X = np.concatenate((x0, X))
    # cập nhật lại kích thước của input X
    N = X.shape[1]
    d = X.shape[0]

    print(X.shape)
    print(y.shape)
    # Chọn ngẫu nhiên một vector hệ số  w với các phần tử gần 0
    w_init = np.random.randn(d, 1)

    # thêm vector ngẫu nhiên vừa khởi tạo w_init vào trong mảng các w
    w = [w_init]

    # điểm dữ liệu đầu vào x bị phân loại sai lớp
    mis_points = []

    # sử dụng while true : chuẩn
    # while True:
    #  vì dữ liệu quá lớn nên chạy vòng có giới hạn để cần phải trả về kqua kiểm tra
    for repeat in range(10000):
        # mix data
        # xáo trộn dữ liệu đảm bảo tính ngẫu nhiên của dữ liệu
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


# hàm chuyển đầu ra của class y thành -1 và 1
def formart_class(y):
    y_new = []
    for i in y:
        if (i == 2):
            y_new.append(-1)
        else:
            y_new.append(1)
    result = np.array(y_new)
    return result

def formart_class_zero_one(y):
    y_new = []
    for i in y:
        if (i == -1):
            y_new.append(0)
        elif(i == 1):
            y_new.append(1)
        else:
            return None
    result = np.array(y_new)
    return result

if __name__ == '__main__':
    url_data = "./data/data.txt"
    data, X_train, y_train, X_test, y_test = read_data.read_data(url_data)

    # formart result => chuyển thành tập đích là -1 và 1
    y_train = formart_class(y_train)
    y_test = formart_class(y_test)

    d = X_train.shape[0]

    print("Perceptron: ")
    # (w, m) = perceptron(X_train, y_train)
    # print(w[-1])

    # sau khi trainning sinh ra tập w
    w = [[-420.36487644], [32.54019542], [3.36617903], [12.34221744], [1.57922832], [-2.03037672], [16.14436801],
         [17.15761069], [-11.21454204], [27.2693682]]
    w = np.array(w)
    y_pred = h(w, X_test).astype(int)
    # làm phẳng ma trận y vì đầu ra ma trận y bị thành vector cột
    y_pred = y_pred.flatten()
    print("y_pred: ")
    print(y_pred)
    print("y_origin: ")
    print(y_test)

    # độ chính xác của mô hình
    accuracy = evaluate_model.accuracy(y_true=y_test, y_pred=y_pred)
    print("accuracy = " + str(accuracy * 100) + "%")

    # vì trong bài toán này đầu ra chúng ta có 2 lớp -1 và 1 => chuyển thành 0 và 1 tương ứng để đưa vào confusion matrix
    y_test = formart_class_zero_one(y_test)
    y_pred = formart_class_zero_one(y_pred)

    (precision, recall) = evaluate_model.cm2pr_binary(y_true=y_test, y_pred=y_pred)
    print("precision = " + str(precision))
    print("recall = " + str(recall))
