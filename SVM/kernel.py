import numpy as np
import random
from SMO import opt, clip_alpha, load_data


class optStruct(opt):
    """
    数据结构（继承自opt）
    """
    def __init__(self, data, label, C, toler, k_tup):
        super().__init__(data, label, C, toler)
        self.K = np.mat(np.zeros((self.m, self.m)))

        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], k_tup)


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
    oS = optStruct(data, label, C, toler, k_tup)
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


def kernel_trans(X, A, k_tup):
    """
    核函数
    计算 X 中所有数据与 A 的核映射
    :param X: [None None] 所有输入数据
    :param A: [None] 待计算数据
    :param k_tup: string 核函数
    :return: [None] 核映射
    """
    m, n = X.shape
    K = np.mat(np.zeros((m, 1)))
    if k_tup[0] == "lin":
        K = X * A.T
    elif k_tup[0] == "rbf":
        for j in range(m):
            delta_row = X[j, :] - A
            K[j] = delta_row * delta_row.T
        K = np.exp(K / (-1 * k_tup[1]**2))
    else:
        raise NameError("Kernel is not recognized")
    return K


def calcEk(oS, k):
    """
    计算偏差 E_i
    :param oS: opt 数据结构
    :param k: int 索引
    :return: float E_k
    """
    f_xk = float(np.multiply(oS.alphas, oS.label).T * oS.K[:, k]) + oS.b
    E_k = f_xk - float(oS.label[k])
    return E_k


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

        eta = float((oS.K[i, i] + oS.K[j, j] \
                     - 2.0 * oS.K[i ,j]))
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
             - float(oS.label[i]) * (oS.alphas[i][0] - alpha_i_old) * (oS.K[i, i]) \
             - float(oS.label[j]) * (oS.alphas[j][0] - alpha_j_old) * (oS.K[j ,i])
        b2 = oS.b - Ej \
             - float(oS.label[j]) * (oS.alphas[j][0] - alpha_j_old) * (oS.K[j ,j]) \
             - float(oS.label[i]) * (oS.alphas[j][0] - alpha_j_old) * (oS.K[i, j])

        if 0 < oS.alphas[i][0] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j][0] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


if __name__ == "__main__":
    data, label = load_data("../data/SVM/testSetRBF.txt")
    data, label = np.mat(data), np.mat(label).reshape(100, 1)
    b, alphas = smo(data, label, 200, 0.0001, 10000, ("rbf", 1.3))

    sv_ind = np.nonzero(alphas.A>0)[0]
    svs, label_sv = data[sv_ind], label[sv_ind]

    # 统计错误率
    m, error = data.shape[0], 0
    for i in range(m):
        kernel_val = kernel_trans(svs, data[i, :], ("rbf", 1.3))
        predict = kernel_val.T * np.multiply(label_sv, alphas[sv_ind]) + b

        if float(predict) * float(label[i]) < 0:
            error += 1
            print("error index: {}".format(i))

    print("error: {}".format(error / float(m)))
