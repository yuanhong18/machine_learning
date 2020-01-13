import  numpy as np
import matplotlib.pyplot as plt

from common.data_process import load_data


def linear_regression(data, label):
    """
    标准线性回归
    损失函数为 SUM (Yi - Xi * W)^2
    W = (X.T*X)-1*X.T*Y
    :param data: [None None] 输入特征
    :param label: [None] 标注
    :return: [None] 权重 W
    """
    if np.linalg.det(data.T * data) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    W = (data.T * data).I * (data.T * label)
    return W


def lwlr(test_point, data, label, k=1.0):
    """
    局部加权线性回归
    W = (X.T*w*X)-1*X.T*w*Y
    :param test_point: 预测点
    :param data: 输入贴纸
    :param label: 标注
    :param k: 高斯核 k 值
    :return:
    """
    m = data.shape[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff = test_point - data[j, :]
        weights[j ,j] = np.exp(diff*diff.T/(-2.0*k**2))
    if np.linalg.det(data.T*(weights*data)) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    W = (data.T*(weights*data)).I * (data.T * (weights * label))
    return test_point * W


def lwlr_test(test_arr, data, label, k=1.0):
    """
    预测
    :param test_point: 待预测点集
    :param data: 输入特征
    :param label: 标注
    :param k: 高斯核 k 值
    :return:
    """
    m = test_arr.shape[0]
    y = np.zeros((m, 1))
    for i in range(m):
        y[i][0] = lwlr(test_arr[i], data, label, k)
    return y


def plot(data, label, Y):
    """
    做出散点图
    :param data: [None None] 散点
    :param label: [None] 标签
    :param W: [None] 预测值
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 1].flatten().A[0], label[:, 0].flatten().A[0], s=5)

    x_copy = data.copy()
    srt_ind = x_copy[:, 1].argsort(0)

    ax.plot(x_copy[srt_ind, 1], Y[srt_ind, 0])
    plt.show()


if __name__ == "__main__":
    data, label = load_data("../data/regression/ex0.txt")
    data, label = np.mat(data), np.mat(label).reshape((len(data), 1))

    W = linear_regression(data, label)
    Y = data * W
    Y_1 = lwlr_test(data, data, label, 0.01)

    plot(data, label, Y)
    plot(data, label, Y_1)
