import os
import numpy as np

url_data = "./data/data.txt"


# count row of data have col 1 = 2
def count_column(data):
    count = 0
    for i in data:
        if (i[1] == '2'):
            count = count + 1
    return count


# sort data by column
def sort_data(data, columnIndex):
    sortedArr = data[data[:, columnIndex].argsort()]
    return sortedArr


def read_data(url):
    with open(url, 'r') as f:
        l_strip = [s.strip().split(',') for s in f.readlines()]
    data = np.array(l_strip)

    # sắp xếp lại cột 3 của data => tạo thành 2 ở trên 4 ở dưới
    sorted_data = sort_data(data, 1)
    count = count_column(sorted_data)
    # print('Số người benign: ', count)
    row_data, col_data = data.shape

    # benign
    benign = sorted_data[0:count, :]  # index = [0, 457]
    row_benign, col_benign = benign.shape

    # malignant
    malignant = sorted_data[count: row_data, :]  # index = [458, 698]
    row_malignant, col_malignant = malignant.shape

    # làm đến đây , kiểm tra phần dưới phần chỉ số đúng chưa trả về kqua la xong

    # train data
    benign_train = benign[0: row_benign - 80, :]
    # print("benign_train.shape: ", benign_train.shape)
    malignant_train = malignant[0: row_malignant - 40, :]
    # print("malignant_train.shape: ", malignant_train.shape)

    # test data
    benign_test = benign[row_benign - 80: row_benign, :]
    # print("benign_test.shape: ", benign_test.shape)
    malignant_test = malignant[row_malignant - 40: row_malignant, :]
    # print("malignant_test.shape: ", malignant_test.shape)

    train_data = np.concatenate((benign_train, malignant_train))
    # print("train_data.shape: ", train_data.shape)

    test_data = np.concatenate((benign_test, malignant_test))
    # print("test_data.shape: ", test_data.shape)
    X_train = train_data[:, 2: row_data + 1].astype(np.float)
    y_train = train_data[:, 1].astype(np.float)

    # test data
    X_test = test_data[:, 2: row_data + 1].astype(np.float)
    y_test = test_data[:, 1].astype(np.float)
    return data, X_train, y_train, X_test, y_test