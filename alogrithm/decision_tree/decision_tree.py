import math
import operator
import pickle


def calculate_shannon_entropy(data_set):
    """
    计算香浓信息熵
    """
    # 获取数据集数据条目的数量
    num_entries = len(data_set)
    labels_count = {}
    # 对数据集中的数据包含的类型数目进行统计
    for feat_vec in data_set:
        label = feat_vec[-1]
        labels_count[label] = labels_count.get(label, 0) + 1
    # 计算香浓信息熵
    shannon_entropy = 0.0
    for key in labels_count.keys():
        prob = labels_count[key] / num_entries
        shannon_entropy -= prob * math.log(prob, 2)
    return shannon_entropy


def split_data_set(data_set, axis, value):
    """
    划分数据集
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[axis + 1 :])
            ret_data_set.append(reduce_feat_vec)
    return ret_data_set


def choose_best_feature(data_set):
    """
    通过衡量信息熵的变化，来选择最佳的分类特征
    """
    # 获取待选的特征的数量，data_set 最后一列为分类
    number_of_feature = len(data_set[0]) - 1
    # 计算当前数据集的香农信息熵，作为基准
    base_shannon_entropy = calculate_shannon_entropy(data_set)
    # 声明初始的信息增益为 0；最好的特征为索引 -1（无效）
    best_info_gain = 0.0
    best_feature = -1
    # 遍历每一个特征
    for i in range(number_of_feature):
        # 获取当前特征的在数据集中的所有数据
        feature_values = [example[i] for example in data_set]
        # 获取当前特征的所有取值
        feature_values_set = set(feature_values)
        new_shannon_entropy = 0
        # 遍历当前特征的所有取值，获取按照当前特征划分后的所有数据集，并计算划分后的香农信息熵
        for feature_value in feature_values_set:
            # 获取按照当前特征值划分后的数据集
            reduced_data_set = split_data_set(data_set, axis=i, value=feature_value)
            # 计算取到当前特征值划分后的数据集的概率
            prob = len(reduced_data_set) / len(data_set)
            # 计算划分数据集后新的信息熵
            new_shannon_entropy += prob * calculate_shannon_entropy(reduced_data_set)
        # 计算按照当前特征划分后的信息增益
        info_gain = base_shannon_entropy - new_shannon_entropy
        # 更新信息增益和最优划分特征
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_class_count = sorted(
        class_count.items(), key=operator.itemgetter(1), reverse=True
    )
    return sorted_class_count[0][0]


def create_decision_tree(data_set, feature_labels):
    class_list = [data[-1] for data in data_set]
    # 如果当前数据子集中数据分类类型相同，退出递归
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果耗尽了所有的特征，对分类进行投票
    if len(data_set[0]) == 1:
        return majority_count(class_list)
    # 首先找到本次划分所基于的特征（索引）
    best_feature_index = choose_best_feature(data_set)
    best_feature_label = feature_labels[best_feature_index]
    decision_tree = {best_feature_label: {}}
    feature_values = [data[best_feature_index] for data in data_set]
    feature_values_set = set(feature_values)
    del feature_labels[best_feature_index]
    for value in feature_values_set:
        sub_labels = feature_labels[:]
        decision_tree[best_feature_label][value] = create_decision_tree(
            split_data_set(data_set, best_feature_index, value), sub_labels
        )
    return decision_tree


def classify(decision_tree, feature_labels, feature_vector):
    root_feature_label = list(decision_tree.keys())[0]
    sub_decision_tree = decision_tree[root_feature_label]
    feature_index = feature_labels.index(root_feature_label)
    for key in sub_decision_tree.keys():
        if feature_vector[feature_index] == key:
            if type(sub_decision_tree[key]).__name__ == "dict":
                class_label = classify(
                    sub_decision_tree[key], feature_labels, feature_vector
                )
            else:
                class_label = sub_decision_tree[key]
    return class_label


def store_decision_tree(decision_tree, filename):
    fw = open(filename, "wb")
    pickle.dump(decision_tree, fw)
    fw.close()


def load_decision_tree(filename):
    fr = open(filename, "rb")
    return pickle.load(fr)


def create_data_set():
    data_set = [
        [1, 1, "yes"],
        [1, 1, "yes"],
        [1, 0, "no"],
        [0, 1, "no"],
        [0, 1, "no"],
    ]
    labels = ["no surfacing", "flippers"]
    return data_set, labels


if __name__ == "__main__":
    data_set, labels = create_data_set()
    decision_tree = create_decision_tree(data_set, labels[:])
    filename = "./decision_tree.txt"
    store_decision_tree(decision_tree, filename)
    print(load_decision_tree(filename))
    # label1 = classify(decision_tree, labels, [1, 0])
    # label2 = classify(decision_tree, labels, [1, 1])
    # print(label1)
    # print(label2)
