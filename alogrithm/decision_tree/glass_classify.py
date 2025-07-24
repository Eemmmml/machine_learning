import math
import operator
import pickle


def calculate_shannon_entropy(data_set):
    """
    计算数据集的香农熵
    """
    number_of_data = len(data_set)
    labels_vector = [data[-1] for data in data_set]
    labels_count_dict = {}
    for label in labels_vector:
        labels_count_dict[label] = labels_count_dict.get(label, 0) + 1
    shannon_entropy = 0.0
    for key in labels_count_dict.keys():
        prob = labels_count_dict[key] / number_of_data
        shannon_entropy -= prob * math.log(prob, 2)
    return shannon_entropy


def split_data_set(data_set, axis, value):
    """
    划分数据集
    """
    sub_data_set = []
    for data in data_set:
        if data[axis] == value:
            sub_data = data[:axis]
            sub_data.extend(data[axis + 1 :])
            sub_data_set.append(sub_data)
    return sub_data_set


def choose_best_feature(data_set):
    number_of_feature = len(data_set[0]) - 1
    base_shannon_entropy = calculate_shannon_entropy(data_set)
    best_feature = -1
    best_info_gain = 0.0
    for i in range(number_of_feature):
        feature_value_vector = [data[i] for data in data_set]
        feature_value_set = set(feature_value_vector)
        shannon_entropy = 0.0
        for value in feature_value_set:
            sub_data_set = split_data_set(data_set, axis=i, value=value)
            prob = feature_value_vector.count(value) / len(feature_value_vector)
            shannon_entropy += calculate_shannon_entropy(sub_data_set) * prob
        info_gain = base_shannon_entropy - shannon_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_count(class_list):
    class_count_dict = {}
    for vote in class_list:
        class_count_dict[vote] = class_count_dict.get(vote, 0) + 1
    sorted_class_count = sorted(
        class_count_dict.items(), key=operator.itemgetter(1), reverse=True
    )
    return sorted_class_count[0][0]


def create_decision_tree(data_set, feature_labels):
    class_list = [data[-1] for data in data_set]
    if len(class_list) == class_list.count(class_list[0]):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_count(class_list)
    best_feature_index = choose_best_feature(data_set)
    best_feature = feature_labels[best_feature_index]
    decision_tree = {best_feature: {}}
    feature_values_vector = [data[best_feature_index] for data in data_set]
    feature_values_set = set(feature_values_vector)
    del feature_labels[best_feature_index]
    for value in feature_values_set:
        sub_data_set = split_data_set(data_set, best_feature_index, value)
        sub_feature_labels = feature_labels[:]
        decision_tree[best_feature][value] = create_decision_tree(
            sub_data_set, sub_feature_labels
        )
    return decision_tree


def classify(decision_tree, feature_vector, feature_labels):
    now_feature_label = list(decision_tree.keys())[0]
    now_feature_index = feature_labels.index(now_feature_label)
    feature_value_dict = decision_tree[now_feature_label]
    for key in feature_value_dict.keys():
        if key == feature_vector[now_feature_index]:
            if type(feature_value_dict[key]).__name__ == "dict":
                class_label = classify(
                    feature_value_dict[key], feature_vector, feature_labels
                )
            else:
                class_label = feature_value_dict[key]
    return class_label


def store_decision_tree(decision_tree, filename):
    fwb = open(filename, "wb")
    pickle.dump(decision_tree, fwb)
    fwb.close()


def load_decision_tree(filename):
    frb = open(filename, "rb")
    decision_tree = pickle.load(frb)
    frb.close()
    return decision_tree


def create_data_set(filename):
    fr = open(filename)
    lines = fr.readlines()
    data_set = []
    for line in lines:
        data_vector = line.strip().split("\t")
        data_set.append(data_vector)
    labels = ["age", "prescript", "astigmatic", "tear rate"]
    return data_set, labels


if __name__ == "__main__":
    # filename = "./lenses.txt"
    # data_set, labels = create_data_set(filename)
    # print(f"labels: {labels}")
    # print(f"data set: {data_set}")
    # decision_tree = create_decision_tree(data_set, labels)
    # print(f"decision tree: {decision_tree}")
    # store_decision_tree(decision_tree, "./glass_decision_tree.txt")
    print()
