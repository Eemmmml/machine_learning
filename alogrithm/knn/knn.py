import numpy as np
import matplotlib.pyplot as ply
import operator


def createDataSet():
    group = np.array([[20, 3], [15, 5], [18, 1], [5, 17], [2, 15], [3, 20]])
    labels = ["服务策略", "服务策略", "服务策略", "平台策略", "平台策略", "平台策略"]

    return group, labels


def classify(in_x, data, labels, k):
    data_size = data.shape[0]
    diff_mat = np.tile(in_x, (data_size, 1)) - data
    sqrt_mat = diff_mat**2
    sub_distance = sqrt_mat.sum(axis=1)
    distance = sub_distance**0.5
    sorted_distance = np.argsort(distance)
    print(sorted_distance)

    class_dic = {}
    for i in range(k):
        vote_label = labels[sorted_distance[i]]
        class_dic[vote_label] = class_dic.get(vote_label, 0) + 1

    sorted_class_count = sorted(operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


if __name__ == "__main__":
    group, labels = createDataSet()
    # x = [item[0] for item in group[:3]]
    # y = [item[1] for item in group[:3]]
    # ply.scatter(x, y, s=30, c="r", marker="x")
    # x = [item[0] for item in group[3:6]]
    # y = [item[1] for item in group[3:6]]
    # ply.scatter(x, y, s=100, c="b", marker="o")
    # ply.show()
    label = classify([4, 17], group, labels, 3)
    print(label)
