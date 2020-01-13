import os
import argparse
import numpy as np
from memory_profiler import profile
import time
from kNN import create


class kNN(object):
    '''k 邻近算法'''
    def __init__(self, k):
        self.k = k          # k 值

    def foward(self, input, data_x, data_y):
        '''预测标签'''
        # 距离计算
        data_size = data_x.shape[0]
        diff = np.tile(input, (data_size, 1)) - data_x
        sq_diff = diff ** 2
        sq_diff = sq_diff.sum(axis=1)
        distances = sq_diff ** 0.5
        # 距离排序
        sorted_distances = distances.argsort()
        # 分类决策
        count = {}
        for i in range(self.k):
            votelabel = data_y[sorted_distances[i]]
            count[votelabel] = count.get(votelabel, 0) + 1
        sorted_count = sorted(count.items(),
                              key=lambda x:x[1],
                              reverse=True)
        return sorted_count[0][0]


def file2matrix(filename):
    '''导入训练数据至 numpy'''
    with open(filename, "r", encoding="utf-8") as f_r:
        array_lines = f_r.readlines()
    num = len(array_lines)
    matrix = np.zeros((num, 3), dtype=float)
    labels = np.zeros((num,), dtype=float)

    index = 0
    for line in array_lines:
        line = line.strip().split()
        matrix[index, :] = line[0: 3]
        labels[index] = line[3]
        index += 1
    return matrix, labels


def auto_norm(dataset):
    '''数据归一化'''
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    num = dataset.shape[0]
    ranges = max_vals - min_vals
    norm_dataset = np.zeros(shape=dataset.shape)
    norm_dataset = (dataset - np.tile(min_vals, (num, 1)))\
                   / np.tile(ranges, (num, 1))
    return norm_dataset


def img2vector(filename):
    '''for minist'''
    vector = np.zeros((1, 1024), dtype=int)    # 默认为 float64
    with open(filename, "r", encoding="utf-8") as f_r:
        lines = f_r.readlines()
    for i, line in enumerate(lines):
        for j in range(32):
            vector[0, 32*i+j] = int(line[j])
    return vector


@profile
def date_predict0(arg):
    '''约会预测'''
    # 读取数据
    knn = kNN(1)
    data_x, data_y = file2matrix(arg.data_path)
    data_x = auto_norm(data_x)
    # 分类输出
    start = time.time()
    error_count = 0
    num_test = int(0.1 * data_x.shape[0])
    for i in range(num_test):
        label = knn.foward(data_x[i, :], data_x[num_test:, :], data_y[num_test:])
        if label != data_y[i]:
            error_count += 1
    end = time.time()

    print("%f seconds" % (end - start))
    print("%f" % (error_count / float(num_test)))


@profile
def data_predict1(arg):
    '''约会预测'''
    # 读取数据
    k = 1
    data_x, data_y = file2matrix(arg.data_path)
    data_x = auto_norm(data_x)
    # 分类输出
    num_test = int(0.1 * data_x.shape[0])
    root = create(data_x[num_test:, :].tolist(), data_y[num_test:].tolist())
    # root = create([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]], [0,1,1,1,0,1])
    start = time.time()
    error_count = 0
    for i in range(num_test):
        res = root.search_knn(data_x[i, :], k)
        # 分类决策
        count = {}
        for j in range(k):
            votelabel = res[j][0].label
            count[votelabel] = count.get(votelabel, 0) + 1
        sorted_count = sorted(count.items(),
                              key=lambda x: x[1],
                              reverse=True)
        if sorted_count[0][0] != data_y[i]:
            error_count += 1
    end = time.time()

    print("%f seconds" % (end - start))
    print("%f" % (error_count / float(num_test)))


@profile
def minist_predict(arg):
    '''手写预测'''
    knn = kNN(10)
    # 导入训练数据
    train_filelist = os.listdir(arg.train_path)
    train_num = len(train_filelist)
    train_data = np.zeros((train_num, 1024))
    train_labels = np.zeros((train_num,))
    for i, file in enumerate(train_filelist):
        file_str = file.split(".")[0]
        label = int(file_str.split("_")[0])
        train_labels[i] = label
        train_data[i, :] = img2vector(os.path.join(arg.train_path, file))
    # 导入测试数据
    test_filelist = os.listdir(arg.test_path)
    error_count = 0
    test_num = len(test_filelist)

    start = time.time()
    for i, file in enumerate(test_filelist):
        file_str = file.split(".")[0]
        label = int(file_str.split("_")[0])
        vector = img2vector(os.path.join(arg.test_path, file))
        predict = knn.foward(vector, train_data, train_labels)
        if label != predict:
            error_count += 1
            print(file_str, "%d %d" % (predict, label))
    end = time.time()

    print("%f seconds" % (end - start))
    print("%f" % (error_count / float(test_num)))


if __name__ == "__main__":
    opt = argparse.ArgumentParser()
    opt.add_argument("--train_path",
                     type=str,
                     default="../data/KNN/trainingDigits",
                     help="path of data")
    opt.add_argument("--test_path",
                     type=str,
                     default="../data/KNN/testDigits",
                     help="path of data")
    opt.add_argument("--data_path",
                     type=str,
                     default="../data/KNN/datingTestSet.txt",
                     help="path of data")
    arg = opt.parse_args()

    date_predict0(arg)
    data_predict1(arg)

    minist_predict(arg)
