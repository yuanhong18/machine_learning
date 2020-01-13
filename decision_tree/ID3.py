from math import log
import copy
import argparse


def calc_shannon_ent(data_set):
    """
    香农定理
    :param data_set: 数据集，最后一维为label
    :return: 熵
    """
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        label = feat_vec[-1]
        label_counts[label] = label_counts.get(label, 0) + 1

    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)

    return shannon_ent


def split_dataset(data_set, axis):
    """
    按指定特征将数据集划分
    :param data_set: 数据集
    :param axis: 指定特征
    :return: 划分后的数据集（特征值：数据子集）
    """
    ret = {}
    for feat_vec in data_set:
        value = feat_vec[axis]
        if value not in ret:
            ret[value] = []
        ret[value].append(feat_vec[:axis] + feat_vec[axis + 1:])

    return ret


def choose_best_feature(dataset):
    """
    选择信息增益最大的feature
    :param dataset: 数据集
    :return: 信息增益最大的feature的id
    """
    if not dataset:
        raise ValueError("dataset is empty.")
    num_feature = len(dataset[0]) - 1

    base_entropy = calc_shannon_ent(dataset)
    best_info_gain, best_feature = 0.0, -1
    for i in range(num_feature):
        ret = split_dataset(dataset, i)

        new_entropy = 0.0
        for feat, items in ret.items():
            prob = len(items) / float(len(dataset))
            new_entropy += prob * calc_shannon_ent(items)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_count(label_list):
    """
    样本最多的类
    :param class_list: 类别列表
    :return: 样本最多的类
    """
    label_count = {}
    for label in label_list:
        label_count = label_count.get(label, 0) + 1

    label_count.sort(key=lambda x: x[1], reverse=True)
    return label_count[0][0]


def create_tree(dataset, labels):
    """
    创建决策树
    :param dataset: 数据集（最后一维度为label）
    :param labels: 标签名称
    :return: 根节点
    """
    label_list = [item[-1] for item in dataset]
    # 属于同一类别
    if label_list.count(label_list[0]) == len(label_list):
        return {label_list[0]: dataset}
    # 特征为空
    if len(dataset[0]) == 1:
        return {majority_count(label_list): dataset}

    # 选择最优属性
    best_feat = choose_best_feature(dataset)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del labels[best_feat]

    # 创建分支节点
    ret = split_dataset(dataset, best_feat)
    for feat, items in ret.items():
        sublabels = copy.deepcopy(labels)
        my_tree[best_feat_label][feat] = create_tree(items, sublabels)

    return my_tree


def ID3(arg):
    with open(arg.data_path, "r", encoding="utf-8") as f_r:
        lenses = [inst.strip().split("\t") for inst in f_r.readlines()]
    labels = ["age", "prescript", "astigmatic", "tearRate"]
    lenses_tree = create_tree(lenses, labels)

    return None


if __name__ == "__main__":
    opt = argparse.ArgumentParser()
    opt.add_argument("--data_path",
                     type=str,
                     default="../data/decision_tree/lenses.txt",
                     help="path of data")
    arg = opt.parse_args()

    ID3(arg)
