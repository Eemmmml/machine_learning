import numpy as np
import matplotlib.pyplot as plt
import operator


def file2matrix(filename):
    """
    读取本地文件中的数据并将数据转化为方便使用的矩阵，和标签向量
    """
    # 打开文件读取数据
    fr = open(filename)
    array_of_lines = fr.readlines()
    # 创建一个行数和数据条目相同，列数和特征数相同的矩阵
    number_of_lines = len(array_of_lines)
    returnMat = np.empty((number_of_lines, 3))
    label_vector = []

    # 遍历每个数据跳目
    for index, line in enumerate(array_of_lines):
        # 去除字符串首尾的空白和换行
        line = line.strip()
        data_list_of_line = line.split("\t")
        # 将数据填入矩阵
        returnMat[index, :] = data_list_of_line[:3]
        label_vector.append(data_list_of_line[-1])

    return returnMat, label_vector


def auto_norm(dataSet):
    """
    对数据集矩阵中的数据进行归一化处理
    norm_value = (value - min) / (max - min)
    """
    # 获取每个特征的最大值和最小值向量
    min_value_vector = np.min(dataSet, axis=0)
    max_value_vector = np.max(dataSet, axis=0)
    # 获取每个特征的极差向量
    range_vector = max_value_vector - min_value_vector
    norm_data_set = np.empty(dataSet.shape)
    m = dataSet.shape[0]
    # 计算归一化的数据
    norm_data_set = dataSet - np.tile(min_value_vector, (m, 1))
    norm_data_set = norm_data_set / np.tile(range_vector, (m, 1))
    return norm_data_set, range_vector, min_value_vector


def test_k_dating_classify():
    """
    测试算法中参数K对于预测错误率的影响
    """
    result = []
    for k in range(1, 500):
        result.append(dating_classify_test(k))

    # 生成横坐标（下标从1开始）
    x_values = [i + 1 for i in range(len(result))]

    # 创建折线图
    plt.figure(figsize=(10, 6))  # 设置图像大小（可选）
    plt.plot(x_values, result, marker="o", linestyle="-", color="b", linewidth=2)

    # 添加标题和标签
    plt.title("Value Changes by Index", fontsize=14)
    plt.xlabel("Index (starting from 1)", fontsize=12)
    plt.ylabel("Value", fontsize=12)

    # 设置横坐标刻度为整数
    plt.xticks(x_values)

    # 添加网格线
    plt.grid(True, linestyle="--", alpha=0.7)

    # 自动调整布局
    plt.tight_layout()

    # 显示图表
    plt.show()


def dating_classify_test(k):
    """
    KNN 算法性能测试
    """
    # 测试比例
    ho_ratio = 0.1
    # 读取数据
    filename = "./datingTestSet.txt"
    dataSet, labels = file2matrix(filename)
    dataSet, _, _ = auto_norm(dataSet)
    number_of_test = int(dataSet.shape[0] * ho_ratio)
    # 统计错误率
    error_count = 0
    for index, i in enumerate(range(number_of_test)):
        class_of_knn_predicate = knn(
            dataSet[i, :], dataSet[number_of_test:, :], labels, k
        )
        print(
            f"TEST OF {index + 1}: The predicate result is {class_of_knn_predicate}; The real result is {labels[i]}"
        )
        if class_of_knn_predicate != labels[i]:
            error_count += 1
    error_rate = error_count / number_of_test
    print(f"The total error rate is {error_rate}")
    return error_rate


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]

    return group, labels


def knn(inX, dataSet, labels, k):
    # 获取数据集
    dataSize = dataSet.shape[0]
    # 将输入数据广播为与数据集相同形状的矩阵
    diffMat = np.tile(inX, (dataSize, 1)) - dataSet
    # 计算样本距离
    squareMat = diffMat**2
    distances = squareMat.sum(axis=1)
    distances = distances**0.5
    # 对样本距离进行排序，同时范围排序的索引列表
    sorted_idx = np.argsort(distances)

    # 对前 k 个元素进行频率统计
    classify_dic = {}
    for i in range(k):
        vote = labels[sorted_idx[i]]
        classify_dic[vote] = classify_dic.get(vote, 0) + 1

    # 找到并返回出现频率最大的类别
    sorted_classify = sorted(
        classify_dic.items(), key=operator.itemgetter(1), reverse=True
    )
    return sorted_classify[0][0]


if __name__ == "__main__":
    dating_classify_test(20)
    # test_k_dating_classify()
