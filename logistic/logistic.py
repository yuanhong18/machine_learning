import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """
    加载数据
    :return: 特征[None, None]，标签[None]
    """
    data, label = [], []
    with open("../data/logistic/testSet.txt", "r", encoding="utf-8") as f_r:
        lines = f_r.readlines()
    for line in lines:
        line_arr = line.strip().split()
        data.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label.append(int(line_arr[2]))
    return data, label


def sigmoid(X):
    """
    sigmiod 函数
    :param X: 输入
    :return: 输出
    """
    return 1.0 / (1 + np.exp(-X))


def grad_ascent(data, label, alpha, iter):
    """
    logistic regression 梯度上升法(极大似然法)
    L'(W) = SUM[Xi * (Y - sigmoid(W * Xi))]
    :param data: 特针输入 X
    :param label: 标注 Y
    :param alpha: 学习率 alpha
    :param iter: 迭代次数
    :return: 权重 weights
    """
    data, label = np.array(data), np.array(label).reshape((len(label), 1))
    weights = np.ones((data.shape[1], 1))

    for k in range(iter):
        h = sigmoid(np.matmul(data, weights))
        error = label - h
        weights = weights + alpha * np.matmul(data.transpose(), error)
    return weights


def stoc_grad_acent(data, label, alpha):
    """
        logistic regression 梯度上升法(极大似然法)
        L'(W) = SUM[Xi * (Y - sigmoid(W * Xi))]
        :param data: 特针输入 X
        :param label: 标注 Y
        :param alpha: 学习率 alpha
        :return: 权重 weights
        """
    data, label = np.array(data), np.array(label).reshape((len(label), 1))
    weights = np.ones((data.shape[1], 1))

    for i in range(200):
        for k in range(data.shape[0]):
            h = sigmoid(np.matmul(data[k], weights))
            error = label[k] - h
            weights = weights + alpha * data[k].reshape(1, 3).transpose() * error.reshape(1, 1)
    return weights


def plot(data, label, weights):
    """
    画出 logistic 回归曲线
    :param data: X1 X2
    :param label: Y
    :param weights: W
    :return: None
    """
    data, label = np.array(data), np.array(label).reshape((len(label), 1))
    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    n = data.shape[0]

    for i in range(n):
        if int(label[i]) == 1:
            xcord1.append(data[i, 1]); ycord1.append(data[i, 2])
        else:
            xcord2.append(data[i, 1]); ycord2.append(data[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c="red", marker="s")
    ax.scatter(xcord2, ycord2, s=30, c="green")
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel("X1"); plt.ylabel("X2")
    plt.show()


if __name__ == "__main__":
    data, label = load_data()
    weights = stoc_grad_acent(data, label, 0.001)
    plot(data, label, weights)
