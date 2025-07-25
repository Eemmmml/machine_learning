import numpy as np
import re
import random


def load_data_set():
    """导入数据集"""
    posting_list = [
        ["my", "dog", "has", "flea", "problems", "help", "please"],
        ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
        ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
        ["stop", "posting", "stupid", "worthless", "garbage"],
        ["mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
        ["quit", "buying", "worthless", "dog", "food", "stupid"],
    ]
    # 1 代表侮辱性文字，0 代表正常言论
    class_vector = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vector


def create_vocabulary_list(data_set):
    """
    创建词汇表(袋)中的词汇单
    """
    vocabulary_set = set([])
    for document in data_set:
        vocabulary_set = vocabulary_set.union(set(document))
    return list(vocabulary_set)


def set_of_data_to_vector(document, vocabulary_list):
    """
    将分词后的文档向量化
    """
    data_vector = [0] * len(vocabulary_list)
    for word in document:
        if word in vocabulary_list:
            # 这里会统计每个词汇出现的数量，是词汇袋的处理方式
            data_vector[vocabulary_list.index(word)] += 1
        else:
            print(f"The vocabulary {word} is not in my vocabulary list.")
    return data_vector


def train_naive_bayes(train_data_set, labels):
    """
    训练朴素贝叶斯分类器需要的参数
    """
    # 计算文档的总数
    number_of_document = len(train_data_set)
    # 计算词汇单中的词汇总数（特征数）
    number_of_vocabulary = len(train_data_set[0])
    # 从数据集中计算文档类型为1的概率(1的含义取决于标注)
    p_abusive = sum(labels) / number_of_document
    # 统计类型0和类型1的词汇向量的初始化（为了避免统计值为0造成的问题这里初始值设为1）
    p0_vector = np.ones(number_of_vocabulary)
    p1_vector = np.ones(number_of_vocabulary)
    # 设置分母的初始值为词汇的总数，保证在平滑时的归一化
    p0_denom = number_of_vocabulary
    p1_denom = number_of_vocabulary
    # 开始统计文档中词汇的出现对文档类型影响的概率值
    for i in range(number_of_document):
        if labels[i] == 1:
            p1_vector += train_data_set[i]
            p1_denom += np.sum(train_data_set[i])
        else:
            p0_vector += train_data_set[i]
            p0_denom += np.sum(train_data_set[i])
    # 为了避免在小数相乘时造成的舍入误差，我们对概率取对数计算加法
    p0_vector = np.log(p0_vector / p0_denom)
    p1_vector = np.log(p1_vector / p1_denom)
    return p0_vector, p1_vector, p_abusive


def classify_by_naive_bayes(document_vector, p1_vector, p0_vector, p_class1):
    """
    朴素贝叶斯分类器
    """
    # 计算当前文档属于分类1和分类0的概率（这个概率没有除以取到当前向量的概率，因为这个都要除相同的数，对比大小没有影响）
    p1 = np.sum(document_vector * p1_vector) + np.log(p_class1)
    p0 = np.sum(document_vector * p0_vector) + np.log(1 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def test_naive_bayes_classify():
    """
    对朴素贝叶斯分类器的初步测试
    """
    posting_list, class_vector = load_data_set()
    vocabulary_list = create_vocabulary_list(posting_list)

    train_data_set = []
    for post in posting_list:
        train_data_set.append(set_of_data_to_vector(post, vocabulary_list))

    p0_vector, p1_vector, p_abusive = train_naive_bayes(train_data_set, class_vector)
    test_entry_1 = ["love", "my", "dalmation"]
    test_entry_2 = ["stupid", "garbage"]
    test_vector_1 = set_of_data_to_vector(test_entry_1, vocabulary_list)
    test_vector_2 = set_of_data_to_vector(test_entry_2, vocabulary_list)
    class_of_document_1 = classify_by_naive_bayes(
        test_vector_1, p1_vector, p0_vector, p_abusive
    )
    class_of_document_2 = classify_by_naive_bayes(
        test_vector_2, p1_vector, p0_vector, p_abusive
    )
    print(class_of_document_1)
    print(class_of_document_2)


def text_parse(text):
    """
    解析文本
    """
    # 以除了字母和数字以外的所有字符来分割文档
    list_of_words = re.split(r"\W", text)
    return [token.lower() for token in list_of_words if len(token) > 2]


def spam_test():
    """
    对贝叶斯分类器进行交叉测试
    """
    document_list = []
    vocabulary_list = []
    class_label_list = []
    for i in range(1, 26):
        with open(f"./email/spam/{i}.txt") as email:
            list_of_words = text_parse(email.read())
            document_list.append(list_of_words)
            vocabulary_list.extend(list_of_words)
            class_label_list.append(1)
        with open(f"./email/ham/{i}.txt") as email:
            list_of_words = text_parse(email.read())
            document_list.append(list_of_words)
            vocabulary_list.extend(list_of_words)
            class_label_list.append(0)
    vocabulary_set = list(set(vocabulary_list))
    train_set = list(range(50))
    test_set = []
    for i in range(10):
        rand_index = random.randint(0, len(train_set) - 1)
        test_set.append(train_set[rand_index])
        del train_set[rand_index]
    train_matrix = []
    train_class_labels = []
    for doc_index in train_set:
        train_matrix.append(
            set_of_data_to_vector(
                document_list[doc_index], vocabulary_list=vocabulary_set
            ),
        )
        train_class_labels.append(class_label_list[doc_index])
    p0_vector, p1_vector, p_rubbish = train_naive_bayes(
        train_matrix, train_class_labels
    )
    err_count = 0
    for doc_index in test_set:
        predicted_class = classify_by_naive_bayes(
            set_of_data_to_vector(
                document_list[doc_index], vocabulary_list=vocabulary_set
            ),
            p1_vector,
            p0_vector,
            p_rubbish,
        )
        print(
            f"The result of predicted is {predicted_class}, the real result is {class_label_list[doc_index]}"
        )
        if predicted_class != class_label_list[doc_index]:
            err_count += 1
    print(f"The error rate is {err_count / 10 * 100}%")


if __name__ == "__main__":
    # test_naive_bayes_classify()
    spam_test()
    # test()
