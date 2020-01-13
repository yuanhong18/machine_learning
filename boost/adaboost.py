import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
    """
    导入数据
    :param file_name: string 文件名
    :return: [None None] [None] 输入数据 标注
    """
    with open(file_name, "r") as f_r:
        lines = f_r.readlines()
    num_feat = len(lines[0].strip().split("\t"))
    data, label = [], []

    for line in lines:
        item = []
        line_arr = line.strip().split("\t")
        for i in range(num_feat - 1):
            item.append(float(line_arr[i]))
        data.append(item)
        label.append(float(line_arr[-1]))
    return data, label


class line(object):
    def __int__(self):
        self.point = 0.
        self.dim = 0
        self.direction = -1

def create_line(x, y , D):
    min_err, inequal = 1., [-1, 1]
    line_ = line()
    num_steps, num_feat = 10, x.shape[1]
    for i in range(num_feat):
        range_min, range_max = data[:, i].min(), data[:, i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, num_steps + 1):
            point = range_min + float(j) * step_size
            for k in range(2):    # -1 分割点左边 -1 右边 1, 1 分割点左边 1 右边-1
                res = np.ones(y.shape) * inequal[k]
                res[x[:, i] > point] = inequal[(k + 1) % 2]
                err = np.sum(D[res != y])
                if err < min_err:
                    min_err = err
                    line_.point = point
                    line_.dim = i
                    line_.direction = inequal[k]
    return line_

def predictbase(line_, x):
    res = np.ones(x.shape[0]) * line_.direction
    res[x[:, line_.dim] > line_.point] = line_.direction * (-1)
    return res

def adaboost_predict(x, lines, weights):
    sum = np.zeros((x.shape[0],))
    for line, alph in zip(lines, weights):
        sum += alph * predictbase(line, x)
    result = np.ones((x.shape[0],))
    result[sum<0] = -1
    return result.astype(int), sum

def adaboost(data, label, T=3):
    """
    adaboost
    :param data: [None None] 输入数据
    :param label: [None] 标签
    :param T: int 迭代次数
    :return: [line] [None] [None] 分类器 分类器权重 数据权重
    """
    # 初始化权重
    D = np.ones(label.shape, dtype=np.float) / label.shape
    # 学习器
    lines = []
    # 学习器权重
    weights = []

    for _ in range(T):
        line_ = create_line(data, label, D)
        label_ = predictbase(line_, data)

        err_rate = np.sum(D[label_ != label])
        if(err_rate > 0.5) | (err_rate == 0.):
            break
        alph = np.log((1. - err_rate) / err_rate) / 2
        lines.append(line_)
        weights.append(alph)
        # 更新每个样本权重
        err_index = np.ones(label.shape)
        err_index[label_==label] = -1
        D = D * np.exp(err_index*alph)
        D = D / np.sum(D)
    return lines, weights, D


def plot_roc(res, label):
    """
    绘制 ROC 曲线
    :param res: 预测结果
    :param label: 标注
    :return: None
    """
    cursor = (1.0, 1.0)
    num_pos = np.sum(label == 1.0)
    y_step, x_step = 1/float(num_pos), 1/float(len(label) - num_pos)
    sorted_indices = res.argsort()

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    for index in sorted_indices:
        if label[index] == 1.0:
            delta_x, delta_y = 0, y_step
        else:
            delta_x, delta_y = x_step, 0
        ax.plot([cursor[0], cursor[0]-delta_x], [cursor[1], cursor[1]-delta_y], "b")
        cursor = (cursor[0] - delta_x, cursor[1] - delta_y)
    ax.plot([0, 1], [0, 1], "b--")
    plt.xlabel("False Positive Rate");plt.ylabel("True Positive Rate")
    ax.axis([0,1,0,1])
    plt.show()


if __name__ == "__main__":
    data, label = load_data("../data/adaboost/horseColicTraining2.txt")
    data, label = np.array(data), np.array(label)
    lines, weights, D = adaboost(data, label, 15)
    result, res = adaboost_predict(data, lines, weights)
    print(result, "\n", label, "\n", D)
    print("acc: {}".format(np.mean(result == label)))

    plot_roc(res, label)
