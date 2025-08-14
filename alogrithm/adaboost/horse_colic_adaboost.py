import numpy as np


def load_data_set(filename):
    data_set = []
    class_labels = []
    with open(filename) as fr:
        for line in fr.readlines():
            data_array = line.strip().split("\t")
            data_set.append([float(data) for data in data_array[:-1]])
            class_labels.append(float(data_array[-1]))
    return data_set, class_labels


def stump_classify(data_matrix, dimen, thresh_value, thresh_in_equal):
    """
    通过单层决策树对输入数据进行分类(二分类)
    dimen: 分类特征的索引
    thresh_value: 分类的阀值
    thresh_in_equal: 分类标准是大于还是小于["lt", "gt"]
    返回分类结果的列向量
    """
    m, _ = np.shape(data_matrix)
    result_vector = np.asmatrix(np.ones((m, 1)))
    if thresh_in_equal == "lt":
        result_vector[data_matrix[:, dimen] <= thresh_value] = -1
    else:
        result_vector[data_matrix[:, dimen] > thresh_value] = -1
    return result_vector


def build_stump(data_matrix, label_vector, d):
    """
    构建单层决策树
    data_set: 数据集
    class_labels: 数据集中数据对应的标签
    d: 每条数据对应的权重，为列向量(经过归一化处理)
    返回一个单层决策树(字典存储相关参数)
    """
    min_weight_error_rate = np.inf
    # 最后返回的存储单层决策树参数的字典
    best_stump = {}
    # 阀值增长的步数
    number_of_step = 10
    m, n = np.shape(data_matrix)
    best_classification_estimator = np.asmatrix(np.ones((m, 1)))
    # 遍历数据的每一个特征选择加权错误率最低的特征
    for i in range(n):
        # 当前特征所有数据值的最小值
        feature_min_value = np.min(data_matrix[:, i])
        # 当前特征所有数据值的最大值
        feature_max_value = np.max(data_matrix[:, i])
        # 计算阀值增长时的步长
        size_of_step = (feature_max_value - feature_min_value) / number_of_step
        # 探测每一个阀值计算每个阀值下的加权错误率
        for j in range(-1, number_of_step + 1):
            thresh_value = feature_min_value + j * size_of_step
            for thresh_in_equal in ["lt", "gt"]:
                predicted_value_vector = stump_classify(
                    data_matrix, i, thresh_value, thresh_in_equal
                )
                # 计算加权错误率
                result_check_vector = np.asmatrix(np.ones((m, 1)))
                result_check_vector[predicted_value_vector == label_vector] = 0
                weight_error_rate = d.T * result_check_vector
                if weight_error_rate < min_weight_error_rate:
                    min_weight_error_rate = weight_error_rate
                    best_stump["dimen"] = i
                    best_stump["thresh_value"] = thresh_value
                    best_stump["thresh_in_equal"] = thresh_in_equal
                    best_classification_estimator = predicted_value_vector.copy()
    return best_stump, min_weight_error_rate, best_classification_estimator


def adaboost_training_decision_tree(data_set, class_labels, max_iter_number=40):
    # 弱分类器列表
    weak_classifer_array = []
    data_matrix = np.asmatrix(data_set)
    label_vector = np.asmatrix(class_labels).T
    m, _ = np.shape(data_matrix)
    # 初始权重向量
    d = np.asmatrix(np.ones((m, 1)) / m)
    # 集成分类器的预测结果
    aggregated_classification_estimator = np.asmatrix(np.zeros((m, 1)))
    for i in range(max_iter_number):
        print(f"===== Classifer {i + 1} =====")
        stump, weight_error_rate, classification_estimator = build_stump(
            data_matrix, label_vector, d
        )
        print(f"D: {d}")
        weak_classifer_array.append(stump)
        # 计算当前弱分类器在集成分类器中的权重alpha
        alpha = (
            0.5 * np.log((1 - weight_error_rate) / max(1e-16, weight_error_rate)).item()
        )
        print(f"Alpha: {alpha}")
        stump["alpha"] = alpha
        # 更新数据集中数据的权重
        expon = np.multiply(-1 * alpha * label_vector, classification_estimator)
        d = np.multiply(d, np.exp(expon))
        # 对更新后的权重进行归一化处理
        d = d / d.sum()
        # 计算当前集成分类器的预测结果
        aggregated_classification_estimator += d.T * classification_estimator
        print(
            f"Aggregated Classification Estimator: {aggregated_classification_estimator}"
        )
        # 计算当前集成分类器的错误率
        error_check = np.multiply(
            np.sign(aggregated_classification_estimator) != label_vector,
            np.ones((m, 1)),
        )
        error_rate = error_check.sum() / m
        print(f"Error Rate: {error_rate}")
        if error_rate == 0:
            break
    return weak_classifer_array


def adaboost_classify(data_to_classify, classifer_array):
    data_matrix = np.asmatrix(data_to_classify)
    m, _ = np.shape(data_matrix)
    aggregated_classification_estimator = np.asmatrix(np.zeros((m, 1)))
    for classifer in classifer_array:
        classification_estimator = stump_classify(
            data_matrix,
            classifer["dimen"],
            classifer["thresh_value"],
            classifer["thresh_in_equal"],
        )
        aggregated_classification_estimator += (
            classifer["alpha"] * classification_estimator
        )
    return np.sign(aggregated_classification_estimator)


def test_adaboost_classifer():
    data_set, class_labels = load_data_set("./horseColicTraining2.txt")
    classifer_array = adaboost_training_decision_tree(
        data_set, class_labels, max_iter_number=50
    )
    error_count = 0
    data_len = len(data_set)
    for i in range(data_len):
        predicted = adaboost_classify(data_set[i], classifer_array)
        if predicted != class_labels[i]:
            print(
                f"Training Data Set: The predicted value is {predicted}, but the real value is {class_labels[i]}"
            )
            error_count += 1
    print(f"Training Data Set: The Error Rate is {error_count / data_len}")
    data_set, class_labels = load_data_set("./horseColicTest2.txt")
    error_count = 0
    data_len = len(data_set)
    for i in range(data_len):
        predicted = adaboost_classify(data_set[i], classifer_array)
        if predicted != class_labels[i]:
            print(
                f"Test Data Set: The predicted value is {predicted}, but the real value is {class_labels[i]}"
            )
            error_count += 1
    print(f"Test Data Set: The Error Rate is {error_count / data_len}")


if __name__ == "__main__":
    test_adaboost_classifer()
