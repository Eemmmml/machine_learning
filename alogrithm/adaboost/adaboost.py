import numpy as np


def load_data_set():
    data_set = [[1, 2.1], [2, 1.1], [1.3, 1], [1, 1], [2, 1]]
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_set, class_labels


def stump_classify(data_matrix, dimen, thresh_value, thresh_in_equal):
    result_vector = np.asmatrix(np.ones((np.shape(data_matrix)[0], 1)))
    if thresh_in_equal == "lt":
        result_vector[data_matrix[:, dimen] <= thresh_value] = -1.0
    else:
        result_vector[data_matrix[:, dimen] > thresh_value] = -1.0
    return result_vector


def build_stump(data_matrix, label_vector, d):
    d_vector = np.asmatrix(d)
    m, n = np.shape(data_matrix)
    number_of_steps = 10
    min_weight_error = np.inf
    best_stump = {}
    best_classification_estimator = np.asmatrix(np.ones((m, 1)))
    for i in range(n):
        feature_min_value = np.min(data_matrix[:, i])
        feature_max_value = np.max(data_matrix[:, i])
        size_of_step = (feature_max_value - feature_min_value) / number_of_steps
        for j in range(-1, number_of_steps + 1):
            thresh_value = feature_min_value + (size_of_step * float(j))
            for thresh_in_equal in ["lt", "gt"]:
                predicted_value_vector = stump_classify(
                    data_matrix,
                    dimen=i,
                    thresh_value=thresh_value,
                    thresh_in_equal=thresh_in_equal,
                )
                result_check_vector = np.asmatrix(np.ones((m, 1)))
                result_check_vector[predicted_value_vector == label_vector] = 0
                weight_error = d_vector.T * result_check_vector
                # print(
                #     f"Dimen: {i}, Thresh Value: {thresh_value}, Thresh In Equal: {thresh_in_equal}, Weight Error: {weight_error}"
                # )
                if weight_error < min_weight_error:
                    best_stump["dimen"] = i
                    best_stump["thresh_value"] = thresh_value
                    best_stump["thresh_in_equal"] = thresh_in_equal
                    min_weight_error = weight_error
                    best_classification_estimator = predicted_value_vector.copy()
    return best_stump, min_weight_error, best_classification_estimator


def adaboost_training_decision_stump(data_set, class_labels, iter_number=40):
    data_matrix = np.asmatrix(data_set)
    label_vector = np.asmatrix(class_labels).T
    weak_classifier_array = []
    m, _ = np.shape(data_matrix)
    d = np.asmatrix(np.ones((m, 1)) / m)
    # 记录聚合分类器的分类结果
    aggregated_classification_estimator = np.zeros((m, 1))
    for i in range(iter_number):
        stump, weight_error, classification_estimator = build_stump(
            data_matrix, label_vector, d
        )
        print(f"===== Stump {i + 1} =====")
        print(f"D: {d}")
        print(f"Classification Estimator: {classification_estimator}")
        # 计算alpha值
        alpha = 0.5 * np.log((1 - weight_error) / max(weight_error, 1e-16)).item()
        print(f"alpha: {alpha}")
        stump["alpha"] = alpha
        weak_classifier_array.append(stump)
        # 更新数据权重
        expon = np.multiply(-1 * alpha * label_vector, classification_estimator)
        d = np.multiply(d, np.exp(expon))
        # 归一化处理
        d = d / d.sum()
        aggregated_classification_estimator += alpha * classification_estimator
        print(f"Agg Classification Estimator: {aggregated_classification_estimator}")
        agg_errors = np.multiply(
            np.sign(aggregated_classification_estimator) != label_vector,
            np.ones((m, 1)),
        )
        error_rate = np.sum(agg_errors) / m
        print(f"Error Rate: {error_rate}")
        if error_rate == 0.0:
            break
    return weak_classifier_array, d, aggregated_classification_estimator


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
        print(aggregated_classification_estimator)
    return np.sign(aggregated_classification_estimator)


if __name__ == "__main__":
    data_set, class_labels = load_data_set()
    classifer_array, d, agg_classification_estimator = adaboost_training_decision_stump(
        data_set, class_labels, 9
    )
    # print(classifier_array)
    classify_result = adaboost_classify([0, 0], classifer_array)
    print(classify_result)
    classify_result = adaboost_classify([[5, 5], [0, 0]], classifer_array)
    print(classify_result)
