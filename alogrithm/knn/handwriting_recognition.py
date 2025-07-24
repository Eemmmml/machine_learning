import numpy as np
import operator
import os


def knn_classify(in_x, data_set, labels, k):
    """
    实现 KNN 算法
    """
    number_of_data = data_set.shape[0]
    diff_mut = np.tile(in_x, (number_of_data, 1)) - data_set
    diff_mut **= 2
    diff_mut = np.sum(diff_mut, axis=1)
    distance = diff_mut**0.5
    distance_sorted_index = np.argsort(distance)

    classify_dic = {}
    for i in range(k):
        vote = labels[distance_sorted_index[i]]
        classify_dic[vote] = classify_dic.get(vote, 0) + 1

    sorted_classify_dic = sorted(
        classify_dic.items(), key=operator.itemgetter(1), reverse=True
    )

    return sorted_classify_dic[0][0]


def img2vector(filename):
    fr = open(filename)
    img_vector = np.empty((1, 1024))
    for i in range(32):
        line = fr.readline().strip()
        for j in range(32):
            img_vector[0, i * 32 + j] = int(line[j])
    return img_vector


def handwriting_recognition():
    # 使用训练集构建原始的距离计算矩阵
    # 训练集的文件列表
    training_file_list = os.listdir("./digits/trainingDigits")
    number_of_training_file = len(training_file_list)
    # 向量化后的训练数据集
    training_data_set = np.empty((number_of_training_file, 1024))
    # 训练集的标签向量
    training_labels = []
    for index, training_file in enumerate(training_file_list):
        filename = training_file.split(".")[0]
        label = filename.split("_")[0]
        training_labels.append(label)
        training_data_set[index, :] = img2vector(
            f"./digits/trainingDigits/{training_file}"
        )

    # 使用测试集对算法进行测试
    test_file_list = os.listdir("./digits/testDigits")
    number_of_test_file = len(test_file_list)
    error_count = 0
    for index, test_file in enumerate(test_file_list):
        filename = test_file.split(".")[0]
        real_label = filename.split("_")[0]
        test_img_vector = img2vector(f"./digits/testDigits/{test_file}")
        predicated_label = knn_classify(
            test_img_vector, training_data_set, training_labels, 3
        )
        if real_label != predicated_label:
            print(
                f"TEST {index + 1}: The predicted result is {predicated_label}, The real result is {real_label}"
            )
            error_count += 1

    print(f"The error rate is {error_count / number_of_test_file}")


if __name__ == "__main__":
    handwriting_recognition()
