import typing

import numpy as np


def load_data_set(filename):
    data_set = []
    with open(filename) as fr:
        for line in fr.readlines():
            data_in_line = line.strip().split("\t")
            data_set.append(list(map(float, data_in_line)))
    return data_set


def binary_split_data_set(data_matrix, feature_index, feature_value):
    left_data_matrix = data_matrix[
        np.nonzero(data_matrix[:, feature_index] > feature_value)[0], :
    ]
    right_data_matrix = data_matrix[
        np.nonzero(data_matrix[:, feature_index] <= feature_value)[0], :
    ]
    return left_data_matrix, right_data_matrix


def liner_regression(data_matrix, lam=0.3):
    """
    岭回归版本的线性回归，防止矩阵奇异问题
    """
    m, n = np.shape(data_matrix)
    x = np.asmatrix(np.ones((m, n)))
    y = np.asmatrix(np.ones((m, 1)))
    i = np.asmatrix(np.eye(n))
    x[:, 1:n] = data_matrix[:, : n - 1]
    y = data_matrix[:, -1]
    x_t_x = x.T * x
    demon = typing.cast(np.matrix, x_t_x + i * lam)
    # demon = x_t_x + i * lam
    if np.linalg.det(demon) == 0:
        raise NameError(
            "This matrix is singlar, cannot inverse.\nplease choose another opt."
        )
    w = demon.I * x.T * y
    return w, x, y


# def liner_regression(data_matrix, lam=0.3):
#     m, n = np.shape(data_matrix)
#     x = np.asmatrix(np.ones((m, n)))
#     y = np.asmatrix(np.ones((m, 1)))
#     x[:, 1:n] = data_matrix[:, : n - 1]
#     y = data_matrix[:, -1]
#     x_t_x = x.T * x
#     # demon = x_t_x + i * lam
#     if np.linalg.det(x_t_x) == 0:
#         raise NameError(
#             "This matrix is singlar, cannot inverse.\nplease choose another opt."
#         )
#     w = x_t_x.I * x.T * y
#     return w, x, y


def liner_regression_error(data_matrix):
    w, x, y = liner_regression(data_matrix)
    y_hat = x * w
    return np.sum(np.power(y_hat - y, 2))


def choose_best_split_way(
    data_matrix,
    leaf_type=liner_regression,
    error_type=liner_regression_error,
    opt=(1, 4),
):
    if len(set(data_matrix[:, -1].T.tolist()[0])) == 1:
        print("It is a leaf node, no need to split.")
        return None, leaf_type(data_matrix)[0]
    _, n = np.shape(data_matrix)
    toler_n = opt[1]
    toler_error = opt[0]
    best_error = np.inf
    best_feature_index = 0
    best_feature_value = 0
    error = error_type(data_matrix)
    for feature_index in range(n - 1):
        for feature_value in set(data_matrix[:, feature_index].T.tolist()[0]):
            print(f"Feature Index: {feature_index}, Feature Value: {feature_value}")
            left_data_matrix, right_data_matrix = binary_split_data_set(
                data_matrix, feature_index, feature_value
            )
            if (np.shape(left_data_matrix)[0] < toler_n) or (
                np.shape(right_data_matrix)[0] < toler_n
            ):
                print(
                    f"Toler N: {toler_n}, Left Matrix: {np.shape(left_data_matrix)[0]}, Right Matrix: {np.shape(right_data_matrix)[0]}"
                )
                print("The split is too small.")
                continue
            new_error = error_type(left_data_matrix) + error_type(right_data_matrix)
            print(f"New Error: {new_error}, Best Error: {best_error}")
            if new_error < best_error:
                best_error = new_error
                best_feature_index = feature_index
                best_feature_value = feature_value
    if error - best_error < toler_error:
        print(f"Error: {error}, Best Error: {best_error}")
        print("The error decrease is to small, no need to split.")
        return None, leaf_type(data_matrix)[0]
    left_data_matrix, right_data_matrix = binary_split_data_set(
        data_matrix, best_feature_index, best_feature_value
    )
    if (np.shape(left_data_matrix)[0] < toler_n) or (
        np.shape(right_data_matrix)[0] < toler_n
    ):
        print("One matrix is too small, no need to split.")
        return None, leaf_type(data_matrix)[0]
    return best_feature_index, best_feature_value


def create_model_tree(
    data_matrix,
    leaf_type=liner_regression,
    error_type=liner_regression_error,
    opt=(1, 4),
):
    feature_index, feature_value = choose_best_split_way(
        data_matrix, leaf_type, error_type, opt
    )
    if feature_index == None:
        return feature_value
    left_data_matrix, right_data_matrix = binary_split_data_set(
        data_matrix, feature_index, feature_value
    )
    model_tree = {}
    model_tree["feature_index"] = feature_index
    model_tree["feature_value"] = feature_value
    model_tree["left_tree"] = create_model_tree(
        left_data_matrix, leaf_type, error_type, opt
    )
    model_tree["right_tree"] = create_model_tree(
        right_data_matrix, leaf_type, error_type, opt
    )
    return model_tree


def regression_tree_evaluation(leaf_node, data_to_evaluation):
    return float(leaf_node)


def model_tree_evaluation(leaf_node, data_to_evaluation):
    _, n = np.shape(data_to_evaluation)
    x = np.asmatrix(np.ones((1, n + 1)))
    x[:, 1 : n + 1] = data_to_evaluation
    return float(x * leaf_node)


def is_tree(tree_node):
    return type(tree_node).__name__ == "dict"


def tree_fore_cast(tree, data_to_evaluation, evaluation_type):
    if not is_tree(tree):
        return evaluation_type(tree, data_to_evaluation)
    if data_to_evaluation[tree["feature_index"]] > tree["feature_value"]:
        if is_tree(tree["left_tree"]):
            return tree_fore_cast(
                tree["left_tree"], data_to_evaluation, evaluation_type
            )
        else:
            return evaluation_type(tree["left_tree"], data_to_evaluation)
    else:
        if is_tree(tree["right_tree"]):
            return tree_fore_cast(
                tree["right_tree"], data_to_evaluation, evaluation_type
            )
        else:
            return evaluation_type(tree["right_tree"], data_to_evaluation)


def create_evaluation(tree, matrix_to_evaluation, evaluation_type):
    m, _ = np.shape(matrix_to_evaluation)
    y = np.asmatrix(np.ones((m, 1)))
    for i in range(m):
        y[i, 0] = tree_fore_cast(tree, matrix_to_evaluation[i, :], evaluation_type)
    return y


if __name__ == "__main__":
    filename = "./bikeSpeedVsIq_train.txt"
    opt = (2, 20)
    data_matrix = np.asmatrix(load_data_set(filename))
    tree = create_model_tree(data_matrix, opt=opt)
    filename = "./bikeSpeedVsIq_test.txt"
    data_matrix = np.asmatrix(load_data_set(filename))
    y_hat = create_evaluation(
        tree, data_matrix[:, :-1], evaluation_type=model_tree_evaluation
    )
    correlation_coefficient = np.corrcoef(y_hat.T, data_matrix[:, -1], rowvar=False)[
        0, 1
    ]
    print(correlation_coefficient)
