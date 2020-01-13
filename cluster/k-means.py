import copy as cp
import random as rd
import matplotlib.pyplot as plt

'''
@author :chenyuqing
@mail   :chen_yu_qin_g@163.com
'''
from numpy import *


def load_data(path):
    '''
    :param path:传递路径,返回样例的数据
    :return:
    '''
    data_set = []
    file_object = open(path)
    for line in file_object.readlines():
        lineArr = line.strip().split()
        lineArr = [float(x) for x in lineArr]  # 将字符串转换成数字
        data_set.append(lineArr)
    data_set = array(data_set)
    return data_set


def my_kmeans(k, data_set):
    '''
    :param k:
    :param data_set:
    :return:
    '''
    # sample_data_index = rd.sample(list(range(0, len(data_set))), k)
    sample_data_index = [5, 10, 15]
    start_list = []  # 定义起始的结果向量
    end_list = [[0, 0] for n in range(k)]  # 定义结束的向量
    end_result = [[] for n in range(k)]  # 分类完毕后的结果
    for temp in sample_data_index:
        start_list.append(data_set[temp].tolist())

    iter_n = 10
    while (start_list != end_list):
        for i in range(0, len(data_set)):
            temp_distance = float("inf")
            temp_result = 0
            for j in range(0, len(start_list)):
                distance = math.sqrt(
                    math.pow(data_set[i][0] - start_list[j][0], 2) + math.pow(data_set[i][1] - start_list[j][1], 2))
                if distance < temp_distance:
                    temp_distance = distance
                    temp_result = j  # 明确该点是属于哪一个类别
            end_result[temp_result].append(data_set[i].tolist())
        end_list = cp.deepcopy(start_list)
        for i in range(0, len(end_result)):
            start_list[i][0] = round(sum([x[0] for x in end_result[i]]) / float(len(end_result[i])), 6)  # 注意这里保留小数，不然会死循环，因为拷贝的时候也有精度误差。
            start_list[i][1] = round(sum([x[1] for x in end_result[i]]) / float(len(end_result[i])), 6)
    print("the result is :\n", end_result)
    return end_result


if __name__ == '__main__':
    print("------------my kmeans-----------")
    path = u"./西瓜数据集4.0.txt"
    data_set = load_data(path=path)
    print(data_set)
    result = my_kmeans(3, data_set=data_set)
    print(result[0])
    print(result[1])
    print(result[2])

    one_x = [x[0] for x in result[0]]
    one_y = [x[1] for x in result[0]]

    two_x = [x[0] for x in result[1]]
    two_y = [x[1] for x in result[1]]

    three_x = [x[0] for x in result[2]]
    three_y = [x[1] for x in result[2]]

    plt.scatter(one_x, one_y, s=20, marker='o', color='m')
    plt.scatter(two_x, two_y, s=20, marker='+', color='c')
    plt.scatter(three_x, three_y, s=20, marker='*', color='r')
    plt.show()
