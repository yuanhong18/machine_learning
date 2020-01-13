import numpy as np

def load_data(file_name):
    """
    导入数据
    :param file_name: string 文件名
    :return: [None None] [None] 数据及标注
    """
    with open(file_name, "r") as f_r:
        lines = f_r.readlines()

    data, label = [], []
    for line in lines:
        line_arr = line.strip().split("\t")
        data.append([float(line_arr[i]) for i in range(len(line_arr) - 1)])
        label.append(float(line_arr[-1]))
    return data, label
