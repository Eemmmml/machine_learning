import numpy as np


def load_data_set(filename):
    """
    加载数据集
    filename: 保存数据的.txt文件的文件名
    """
    data_set = []
    with open(filename) as fr:
        for line in fr.readlines():
            data_in_line = line.strip().split("\t")
            data_set.append(list(map(float, data_in_line)))
    return data_set


def binary_split_data_set(data_matrix, feature_index, feature_value):
    """
    对数据集进行二元分割
    data_matrix: 保存了数据的矩阵
    feature_index: 分割数据集的特征的索引
    feature_value: 分割数据集的特征的值
    """
    data_matrix1 = data_matrix[
        np.nonzero(data_matrix[:, feature_index] > feature_value)[0], :
    ]
    data_matrix2 = data_matrix[
        np.nonzero(data_matrix[:, feature_index] <= feature_value)[0], :
    ]
    return data_matrix1, data_matrix2


def regression_leaf(data_matrix):
    """
    计算叶子节点的值，作为回归结果
    data_matrix: 划分到叶子节点中的数据组成的矩阵
    这里将划分到叶子节点的数据的真实值的均值作为回归结果
    """
    return np.mean(data_matrix[:, -1])


def regression_error(data_matrix):
    """
    计算节点的不纯度(残差平方和)
    data_matrix: 划分到当前节点中的数据组成的矩阵
    这里将划分到当前节点的数据的真实值的均值作为回归结果
    """
    return np.var(data_matrix[:, -1]) * np.shape(data_matrix)[0]


def choose_best_split_way(
    data_matrix, leaf_type=regression_leaf, error_type=regression_error, opt=(1, 4)
):
    """
    选择最优的特征和特征值用来划分数据集
    """
    # 如果当前划分中的所有值相同，则无需进行划分直接返回叶子节点的结果
    if len(set(data_matrix[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data_matrix)
    # opt 中的两个值分别对应最小的集合大小和可以忽略的不纯度下降幅度的最大值
    toler_n = opt[0]
    toler_error = opt[1]
    # 初始化选取的特征和特征值，以及集合的不纯度
    best_error = np.inf
    best_feature_index = 0
    best_feature_value = 0
    _, n = np.shape(data_matrix)
    error = error_type(data_matrix)
    # 遍历每个特征和特征值，通过计算不纯度来确定最优的特征和特征值组合
    for feature_index in range(n - 1):
        for feature_value in set(data_matrix[:, feature_index].T.tolist()[0]):
            split_matrix0, split_matrix1 = binary_split_data_set(
                data_matrix, feature_index, feature_value
            )
            if (np.shape(split_matrix0)[0] < toler_n) or (
                np.shape(split_matrix1)[0] < toler_n
            ):
                continue
            new_error = error_type(split_matrix0) + error_type(split_matrix1)
            if new_error < best_error:
                best_error = new_error
                best_feature_index = feature_index
                best_feature_value = feature_value
    # 如果当前划分后的不纯度下降小于可忽略下降幅度的最大值，则可认为当前节点是无需继续划分的叶子节点
    if error - best_error < toler_error:
        return None, leaf_type(data_matrix)
    split_matrix0, split_matrix1 = binary_split_data_set(
        data_matrix, best_feature_index, best_feature_value
    )
    # 如果划分后有一个集合的元素少于最小集合的大小，也放弃划分集合，直接将当前集合作为叶子节点返回
    if (np.shape(split_matrix0)[0] < toler_n) or (np.shape(split_matrix1)[0] < toler_n):
        return None, leaf_type(data_matrix)
    return best_feature_index, best_feature_value


def create_regression_tree(
    data_matrix, leaf_type=regression_leaf, error_type=regression_error, opt=(1, 4)
):
    """
    创建回归树
    """
    feature_index, feature_value = choose_best_split_way(
        data_matrix, leaf_type, error_type, opt
    )
    if feature_index == None:
        return feature_value
    regression_tree = {}
    # 记录当前节点划分子树的特征和特征值
    regression_tree["split_feature_index"] = feature_index
    regression_tree["split_feature_value"] = feature_value
    left_data_matrix, right_data_matrix = binary_split_data_set(
        data_matrix, feature_index, feature_value
    )
    # 递归地创建左右子树
    regression_tree["left_tree"] = create_regression_tree(
        left_data_matrix, leaf_type, error_type, opt
    )
    regression_tree["right_tree"] = create_regression_tree(
        right_data_matrix, leaf_type, error_type, opt
    )
    return regression_tree


def is_tree(regression_tree_node):
    return type(regression_tree_node).__name__ == "dict"


def get_mean(regression_tree):
    if is_tree(regression_tree["left_tree"]):
        regression_tree["left_tree"] = get_mean(regression_tree["left_tree"])
    if is_tree(regression_tree["right_tree"]):
        regression_tree["right_tree"] = get_mean(regression_tree["right_tree"])
    return (regression_tree["left_tree"] + regression_tree["right_tree"]) / 2


def purne(regression_tree, test_data_matrix):
    """
    对回归树进行剪枝
    regression_tree: 由训练集训练出的有些过拟合的回归树
    test_data_matrix: 存储测试集数据的矩阵
    """
    # 检查测试集，如果测试集为空，这里的处理方式是让回归树坍缩成一个叶节点
    if np.shape(test_data_matrix)[0] == 0:
        return get_mean(regression_tree)
    # 检查回归树的当前递归节点的左或右子树是否是树而非叶节点
    if is_tree(regression_tree["left_tree"]) or is_tree(regression_tree["right_tree"]):
        # 对于有左或右子树的情况，我们对左或右子树进行剪枝(递归)
        left_test_data_matrix, right_test_data_matrix = binary_split_data_set(
            test_data_matrix,
            regression_tree["split_feature_index"],
            regression_tree["split_feature_value"],
        )
        # 如果左子树不是叶节点，则对左子树进行剪枝
        if is_tree(regression_tree["left_tree"]):
            regression_tree["left_tree"] = purne(
                regression_tree["left_tree"], left_test_data_matrix
            )
        # 如果右子树不是叶节点，则对右子树进行剪枝
        if is_tree(regression_tree["right_tree"]):
            regression_tree["right_tree"] = purne(
                regression_tree["right_tree"], right_test_data_matrix
            )
    # 对于剪枝后(也有可能左右子树本身就是叶节点没有经过剪枝)
    # 判断当前递归节点的左右子树是否都是叶节点
    if (not is_tree(regression_tree["left_tree"])) and (
        not is_tree(regression_tree["right_tree"])
    ):
        left_test_data_matrix, right_test_data_matrix = binary_split_data_set(
            test_data_matrix,
            regression_tree["split_feature_index"],
            regression_tree["split_feature_value"],
        )
        # 计算不剪枝(不合并左右叶节点)的不纯度
        error_of_no_merge = np.sum(
            np.power(left_test_data_matrix[:, -1] - regression_tree["left_tree"], 2)
        ) + np.sum(
            np.power(right_test_data_matrix[:, -1] - regression_tree["right_tree"], 2)
        )
        # 计算剪枝(合并左右叶节点)的不纯度
        mean_of_current_node = get_mean(regression_tree)
        error_of_merge = np.sum(
            np.power(test_data_matrix[:, -1] - mean_of_current_node, 2)
        )
        # 判断剪枝是否更优
        if error_of_merge < error_of_no_merge:
            print("Merging")
            # 剪枝则返回剪枝后当前递归过程变成叶节点的节点的值
            return mean_of_current_node
        else:
            # 不剪枝则返回以当前递归节点为根节点的回归树
            return regression_tree
    else:
        # 如果左右子树有一个不是叶节点，则直接返回以当前递归的节点为根节点的回归树
        return regression_tree


if __name__ == "__main__":
    data_set = load_data_set("./ex2.txt")
    data_matrix = np.asmatrix(data_set)
    regression_tree = create_regression_tree(data_matrix, opt=(0, 1))
    print(regression_tree)
    test_data_set = load_data_set("./ex2test.txt")
    test_data_matrix = np.asmatrix(test_data_set)
    purned_regression_tree = purne(regression_tree, test_data_matrix)
    print(purned_regression_tree)
