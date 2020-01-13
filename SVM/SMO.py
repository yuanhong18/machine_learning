import random
import numpy as np
import matplotlib.pyplot as plt


def load_data(file):
    """
    导入数据
    :param file: string 数据文件路径
    :return: [None, None] [None] 数据及标注
    """
    data_mat, label_mat = [], []
    with open(file, "r") as f_r:
        lines = f_r.readlines()
    for line in lines:
        line_arr = line.strip().split("\t")
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def selectJ_rand(i,m):
    """
    选择第二个（简化版本） alpha
    :param i: int 第一个 alpha 下标
    :param m: int 所有 alpha 数目
    :return: int 第二个 alpha 下标
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(alpha_j, H, L):
    """
    调整 alpha 值
    L<= alpha_j <=H
    :param alpha_j: float alpha 值
    :param H: float 上界
    :param L: float 下界
    :return: float 调整后的 alpha
    """
    if alpha_j > H:
        alpha_j = H
    if alpha_j < L:
        alpha_j = L
    return alpha_j


def smo_simple(data, label, C, toler, max_iter):
    """
    简化版的 SMO 算法
    :param data: [None, None] 输入特征
    :param label: [None] 标注
    :param C: float 松弛变量
    :param toler: float 容错率
    :param max_iter: int 最大循环次数
    :return: float [None] b alpha
    """
    b, m, n = 0, data.shape[0], data.shape[1]
    alphas = np.zeros((m, 1))
    iter = 0
    while iter < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            f_xi = float(np.multiply(alphas, label).T * (data * data[i, :].T)) + b
            E_i = f_xi - float(label[i])

            if (float(label[i]) * E_i < -toler and alphas[i][0] < C) or \
                    (float(label[i]) * E_i > toler and alphas[i][0] > 0):

                j = selectJ_rand(i, m)      # 随机选择 alpha_j

                f_xj = float(np.multiply(alphas, label).T * (data * data[j, :].T)) + b
                E_j = f_xj - float(label[j])

                alpha_i_old, alpha_j_old = alphas[i][0].copy(), alphas[j][0].copy()

                if float(label[i]) != float(label[j]):
                    L = max(0, float(alphas[j][0]) - float(alphas[i][0]))
                    H = min(C, C + float(alphas[j][0]) - float(alphas[i][0]))
                else:
                    L = max(0, float(alphas[j][0]) + float(alphas[i][0]) - C)
                    H = min(C, float(alphas[j][0]) + float(alphas[i][0]))
                if L == H:
                    print("L == H")
                    continue

                eta = float((data[i, :].dot(data[i, :].T) + data[j, :].dot(data[j, :].T) \
                      - 2.0 * data[i, :] * data[j, :].T))
                if eta <= 0:
                    print("eta <= 0")
                    continue

                alphas[j][0] += float(label[j]) * (E_i - E_j) / eta
                alphas[j][0] = clip_alpha(alphas[j][0], H, L)

                if abs(alphas[j][0] - alpha_j_old) < 0.00001:
                    print("j not moving enough")
                    continue

                alphas[i][0] += float(label[j]) * float(label[i]) * (alpha_j_old - alphas[j][0])
                b1 = b - E_i \
                     - float(label[i]) * (alphas[i][0] - alpha_i_old) * (data[i, :].dot(data[i, :].T)) \
                     - float(label[j]) * (alphas[j][0] - alpha_j_old) * (data[j, :].dot(data[i, :].T))
                b2 = b - E_j \
                     - float(label[j]) * (alphas[j][0] - alpha_j_old) * (data[j, :].dot(data[j, :].T)) \
                     - float(label[i]) * (alphas[j][0] - alpha_j_old) * (data[i, :].dot(data[j, :].T))

                if 0 < alphas[i][0] < C:
                    b = b1
                elif 0 < alphas[j][0] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alpha_pairs_changed += 1
                print("iter: {} i: {} pairs changed: {}".format(iter, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print("iter number: {}".format(iter))
    return b, alphas


class opt:
    """
    用于存储数据的结构
    """
    def __init__(self, data, label, C, toler):
        self.X = data
        self.label = label
        self.C = C
        self.tol = toler
        self.m = data.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))


def calcEk(oS, k):
    """
    计算偏差 E_i
    :param oS: opt 数据结构
    :param k: int 索引
    :return: float E_k
    """
    f_xk = float(np.multiply(oS.alphas, oS.label).T * (oS.X * oS.X[k, :].T)) + oS.b
    E_k = f_xk - float(oS.label[k])
    return E_k


def selectJ(i, oS, Ei):
    """
    选择第二个变量 alpha_j
    首先遍历间隔边界上的点，然后遍历整个数据集，否则返回 inner_layer, 重新选择第一个变量 i
    :param i: int 第一个变量 alpha_i
    :param oS: opt 数据结构
    :param Ei: float 偏差
    :return: int float 第二个变量及偏差 j E_j
    """
    max_K, max_delta_E, E_j = -1, 0, 0
    oS.eCache[i] = [1, Ei]
    valid_Ecache= np.nonzero(oS.eCache[:, 0].A)[0]
    if len(valid_Ecache) > 1:
        for k in valid_Ecache:
            if k == i:  continue
            Ek = calcEk(oS, k)
            delta_E = abs(Ek - Ei)
            if delta_E > max_delta_E:
                max_K, max_delta_E, E_j = k, delta_E, Ek
        return max_K, E_j
    else:
        j = selectJ_rand(i, oS.m)
        E_j = calcEk(oS, j)
    return j, E_j


def update_Ek(oS, k):
    """
    更新差值表 E_i
    :param oS: opt 数据结构
    :param k: int 下标
    :return: None
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def inner_layer(i, oS):
    """
    选择第二个变量，更新 E_i,b,alphas
    :param i: int 第一个变量
    :param oS: opt 数据结构
    :return: int
    """
    Ei = calcEk(oS, i)
    if (oS.label[i][0]*Ei < -oS.tol and oS.alphas[i][0] < oS.C) \
            or (oS.label[i][0]*Ei > oS.tol and oS.alphas[i][0] > 0):
        j, Ej = selectJ(i, oS, Ei)
        alpha_i_old, alpha_j_old = oS.alphas[i][0].copy(), oS.alphas[j][0].copy()

        if oS.label[i][0] != oS.label[j][0]:
            L = max(0, float(oS.alphas[j][0]) - float(oS.alphas[i][0]))
            H = min(oS.C, oS.C + float(oS.alphas[j][0]) - float(oS.alphas[i][0]))
        else:
            L = max(0, float(oS.alphas[j][0]) + float(oS.alphas[i][0]) - oS.C)
            H = min(oS.C, float(oS.alphas[j][0]) + float(oS.alphas[i][0]))
        if L == H:
            print("L == H")
            return 0

        eta = float((oS.X[i, :].dot(oS.X[i, :].T) + oS.X[j, :].dot(oS.X[j, :].T) \
                     - 2.0 * oS.X[i, :] * oS.X[j, :].T))
        if eta <= 0:
            print("eta <= 0")
            return 0

        oS.alphas[j][0] += float(oS.label[j]) * (Ei - Ej) / eta
        oS.alphas[j][0] = clip_alpha(oS.alphas[j][0], H, L)
        update_Ek(oS, j)

        if abs(oS.alphas[j][0] - alpha_j_old) < 0.00001:
            print("j not moving enough")
            return 0

        oS.alphas[i][0] += float(oS.label[j]) * float(oS.label[i]) * (alpha_j_old - oS.alphas[j][0])
        update_Ek(oS, i)

        b1 = oS.b - Ei \
             - float(oS.label[i]) * (oS.alphas[i][0] - alpha_i_old) * (oS.X[i, :].dot(oS.X[i, :].T)) \
             - float(oS.label[j]) * (oS.alphas[j][0] - alpha_j_old) * (oS.X[j, :].dot(oS.X[i, :].T))
        b2 = oS.b - Ej \
             - float(oS.label[j]) * (oS.alphas[j][0] - alpha_j_old) * (oS.X[j, :].dot(oS.X[j, :].T)) \
             - float(oS.label[i]) * (oS.alphas[j][0] - alpha_j_old) * (oS.X[i, :].dot(oS.X[j, :].T))

        if 0 < oS.alphas[i][0] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j][0] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smo(data, label, C, toler, max_iter, k_tup):
    """
    完整 SMO 算法
    :param data: [None, None] 数据源
    :param label: [None, 1] 标注
    :param C: float 松弛变量
    :param toler: float 容错率
    :param max_iter: int 最大迭代次数
    :return: float [None, 1] b,alpha
    """
    oS = opt(data, label, C, toler)
    iter, alpha_pairs_changed = 0, 0
    entire = True
    while iter < max_iter and (alpha_pairs_changed > 0 or entire):
        alpha_pairs_changed = 0
        if entire:
            for i in range(oS.m):
                alpha_pairs_changed += inner_layer(i, oS)
                print("fullset, iter: {} i: {}, pairs changed {}".format(iter, i, alpha_pairs_changed))
            iter += 1
        else:
            non_bound_ids = np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in non_bound_ids:
                alpha_pairs_changed += inner_layer(i, oS)
                print("non-bound, iter: {} i: {}, pairs changed {}".format(iter, i, alpha_pairs_changed))
            iter += 1
        if entire: entire = False
        elif alpha_pairs_changed == 0:  entire = True
        print("iteration number: {}".format(iter))
    return oS.b, oS.alphas


def calc_W(alphas, data, label):
    """
    计算 W
    :param alphas: [None 1] alpha
    :param data: [None None] 数据集
    :param labels: [None] 标注
    :return: [None 1] W
    """
    m, n = data.shape[0], data.shape[1]
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*label[i], data[i, :].T)
    return w


def plot(data, label, weights, b):
    """
    画出 SVM 分类曲线
    :param data: X1 X2
    :param label: Y
    :param weights: W
    :param b: b
    :return: None
    """
    data, label = np.array(data), np.array(label)
    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    n = data.shape[0]

    for i in range(n):
        if int(label[i]) == 1:
            xcord1.append(data[i, 0]); ycord1.append(data[i, 1])
        else:
            xcord2.append(data[i, 0]); ycord2.append(data[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c="red", marker="s")
    ax.scatter(xcord2, ycord2, s=30, c="green")
    x = np.arange(3.0, 6.0, 0.1)
    y = (-float(b) - float(weights[0])*x) / float(weights[1])
    ax.plot(x, y)
    plt.xlabel("X1"); plt.ylabel("X2")
    plt.show()


if __name__ == "__main__":
    data, label = load_data("../data/SVM/testSet.txt")
    data, label = np.mat(data), np.mat(label).reshape(100, 1)
    b, alphas = smo_simple(data, label, 0.6, 0.001, 40)
    w = calc_W(alphas, data, label)

    # 统计错误率
    m, error = data.shape[0], 0
    for i in range(m):
        res = data[i] * np.mat(w) + b

        if float(res) * float(label[i]) < 0:
            error += 1
            print("error index: ".format(i))

    print("error: {}".format(error / float(m)))

    plot(data, label, w, b)
