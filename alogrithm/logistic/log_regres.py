import numpy as np
import random


def load_data_set():
    data_matrix = []
    label_vector = []
    with open("./testSet.txt") as fr:
        for line in fr.readlines():
            data = line.strip().split("\t")
            data_matrix.append([1.0, float(data[0]), float(data[1])])
            label_vector.append(float(data[2]))
    return data_matrix, label_vector


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def grand_ascent(data_set, label_vector):
    """
    基础的梯度上升算法的实现函数
    """
    # 首先将数据集和数据集中的标签向量转化为矩阵（方便后面进行矩阵运算）
    data_set = np.asmatrix(data_set)
    # 将标签向量转化为一个列向量
    label_vector = np.asmatrix(label_vector).transpose()
    # 定义步长（学习率）
    alpha = 0.001
    # 定义学习的最大迭代次数
    max_cycle = 500
    _, n = data_set.shape
    # 定义初始的回归系数都是1
    weights = np.ones((n, 1))
    for _ in range(max_cycle):
        # 这里是梯度上升算法的核心，具体的数学证明可以看笔记的第2节
        h = sigmoid(data_set * weights)
        error = label_vector - h
        weights = weights + alpha * data_set.T * error
    return weights


def scot_grand_ascent(data_set, label_vector):
    """
    随机梯度上升法
    在计算复杂度上做出改进的梯度上升算法，采用在线学习模型
    """
    # 不同于基础版梯度上升算法的实现，这里我们将数据集和标签向量转化为numpy的array
    data_set = np.array(data_set)
    label_vector = np.array(label_vector)
    # 定义学习率
    alpha = 0.01
    m, n = np.shape(data_set)
    weights = np.ones(n)
    for i in range(m):
        # 随机梯度上升算法的核心实现，具体的证明依然见笔记的第2节（矩阵表示之前）
        h = sigmoid(sum(data_set[i] * weights))
        error = label_vector[i] - h
        weights = weights + alpha * data_set[i] * error
    return weights


def better_scot_grand_ascent(data_set, label_vector, iter_num=150):
    """
    改进的随机梯度上升算法
    在改进了基础的梯度上升算法计算复杂度的基础上
    提高了基础随机梯度上升算法存在的算法收敛慢、存在频繁的波动的问题
    """
    data_set = np.array(data_set)
    label_vector = np.array(label_vector)
    m, n = data_set.shape
    weights = np.ones(n)
    for i in range(iter_num):
        data_index = list(range(m))
        for j in range(m):
            # 这里的步长是核心，具体的数学原理目前暂不了解，后续我们会专门学习
            alpha = 4 / (1 + i + j) + 0.01
            rand_index = random.randint(0, len(data_index) - 1)
            h = sigmoid(sum(data_set[rand_index] * weights))
            error = label_vector[rand_index] - h
            weights = weights + alpha * error * data_set[rand_index]
            del data_index[rand_index]
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt

    dataMat, labelMat = load_data_set()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c="red", marker="s")
    ax.scatter(xcord2, ycord2, s=30, c="green")
    x = np.arange(-3.0, 3.0, 0.1)
    y = np.array((-weights[0] - weights[1] * x) / weights[2])
    print(x)
    print(y)
    ax.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


if __name__ == "__main__":
    data_set, label_vector = load_data_set()
    print(len(data_set))
    # weights = grand_ascent(data_set, label_vector)
    # print(weights)
    # weights = scot_grand_ascent(data_set, label_vector)
    # print(weights)
    weights = better_scot_grand_ascent(data_set, label_vector)
    plotBestFit(weights)
