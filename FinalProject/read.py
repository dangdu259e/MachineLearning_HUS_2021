url_data = "./data/data.txt"


def read_data(url):
    file1 = open(url, 'r')
    for line in file1:
        print(file1.readline().strip().split(","))

read_data(url_data)
