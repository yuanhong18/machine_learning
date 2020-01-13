import re
import random
import numpy as np


postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
class_vec = [0,1,0,1,0,1]

def create_vocablist(dataset):
    """
    统计词表
    :param dataset: [None] 句子集合
    :return: [None] 词表
    """
    vocab_set = set()
    for document in dataset:
        vocab_set = vocab_set | set(document)

    return list(vocab_set)


def word_vec(vocab_list, input_set):
    """
    文档向量（贝努力模型，只判断是否出现）
    :param vocab_list: [None] 词汇表
    :param input_set: [None, None] 单个句子
    :return: [None] 句子向量
    """
    vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            vec[vocab_list.index(word)] += 1
        else:
            print("the word: %s is not in vocabulary!" % word)

    return vec


def train_NB0(train_list, train_category):
    """
    计算 P(w|c)
    :param train_list: [None, None] 句子集合（词 id, 贝努力模型）
    :param train_category: [None] 句子类别
    :return:
    """
    num_train = len(train_list)
    num_words = len(train_list[0])

    pabusive = sum(train_category) / float(num_train)
    # p0_num, p1_num = np.zeros(num_words), np.zeros(num_words)
    # p0_sum, p1_sum = 0.0, 0.0

    # 拉普拉斯平滑，避免分母为 0
    p0_num, p1_num = np.ones(num_words), np.ones(num_words)
    p0_sum, p1_sum = 2.0, 2.0
    for i in range(num_train):
        if train_category[i] == 1:
            p1_num += train_list[i]
            p1_sum += sum(train_list[i])
        else:
            p0_num += train_list[i]
            p0_sum += sum(train_list[i])

    p1_vec = p1_num / p1_sum
    p0_vec = p0_num / p0_sum
    return np.log(p0_vec), np.log(p1_vec), pabusive


def classify_NB(vec_classify, p0_vec, p1_vec, pclass):
    """
    分类
    :param vec_classify: [None] 句子向量
    :param p0_vec: [None] P(w|0)
    :param p1_vec: [None] P(w|1)
    :param pclass: float
    :return: int
    """
    p0 = sum(vec_classify * p0_vec) + np.log(pclass)
    p1 = sum(vec_classify * p1_vec) + np.log(pclass)

    if p1 > p0:
        return 1
    else:
        return 0


def testing_NB():
    """
    贝叶斯分类
    :return:
    """
    # 创建词表
    vocab = create_vocablist(postingList)
    # 文本转 id
    train_maxtrix = []
    for post in postingList:
        train_maxtrix.append(word_vec(vocab, post))
    # 计算 P(w|i)
    p0, p1, pa = train_NB0(np.array(train_maxtrix), np.array(class_vec))
    print(p0, p1, pa)

    test_entry = ["love", "my", "dalmation"]
    this = np.array(word_vec(vocab, test_entry))
    print(classify_NB(this, p0, p1, pa))

    test_entry = ["stupid", "garbage"]
    this = np.array(word_vec(vocab, test_entry))
    print(classify_NB(this, p0, p1, pa))


def text_parse(bigstring):
    """
    解析字符串（分词）
    :param bidstring: str
    :return: [None] 分词后的句子
    """
    token_list = re.split(r"\W+", bigstring)
    return [tok.lower() for tok in token_list if len(tok) > 2]


def spam_test():
    """
    垃圾邮件过滤
    :return: None
    """
    # 读取文件
    doc_list, label_list = [], []
    for i in range(1,26):
        with open("../data/bayes/email/spam/%d.txt"%i, "r") as f_r:
            content = f_r.read()
        doc_list.append(text_parse(content))
        label_list.append(1)

        with open("../data/bayes/email/ham/%d.txt"%i, "r") as f_r:
            content = f_r.read()
        doc_list.append(text_parse(content))
        label_list.append(0)

    # 创建词表
    vocab = create_vocablist(doc_list)
    # 划分数据集
    train_list, test_list = list(range(50)), []
    for i in range(10):
        rand_index = int(random.uniform(0, len(train_list)))
        test_list.append(train_list[rand_index])
        del(train_list[rand_index])
    train_matrix, train_label = [], []
    for doc_index in train_list:
        train_matrix.append(word_vec(vocab, doc_list[doc_index]))
        train_label.append(label_list[doc_index])
    # P(w|i)
    p0, p1, psam = train_NB0(np.array(train_matrix), np.array(train_label))
    # 测试结果
    error_count = 0
    for doc_index in test_list:
        ans = word_vec(vocab, doc_list[doc_index])
        if classify_NB(np.array(ans), p0, p1, psam) != label_list[doc_index]:
            error_count += 1
    print(float(error_count) / len(test_list))


if __name__ == "__main__":
    spam_test()
