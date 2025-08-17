import numpy as np


def load_data_set(filename):
    data_set = []
    real_values = []
    with open(filename) as fr:
        for line in fr.readlines():
            features = line.strip().split("\t")
            data_set.append([float(data) for data in features[:-1]])
            real_values.append(float(features[-1]))
    return data_set, real_values


def stand_liner_regression(data_set, real_values):
    """
    计算线性回归的回归系数
    data_set: 类型为列表或numpy数组，为特征值矩阵
    real_values: 类型为列表或numpy数组，为数据集中每条数据对应的实际值
    返回计算后的回归系数
    """
    x = np.asmatrix(data_set)
    y = np.asmatrix(real_values).T
    x_t_x = x.T * x
    if np.linalg.det(x_t_x) == 0:
        print("This matrix is singular, cannot do inverse.")
        return
    w = x_t_x.I * x.T * y
    return w


def locally_weighted_liner_regression(test_point, data_set, real_values, k=1.0):
    """
    计算局部线性回归的回归系数
    test_point: 预测点，其邻近的点将有更大的权重；类型为matrix行向量
    data_set: 类型为列表或numpy数组，为特征值矩阵
    real_values: 类型为列表或numpy数组，为数据集中每条数据对应的实际值
    返回计算后的局部回归系数
    """
    x = np.asmatrix(data_set)
    y = np.asmatrix(real_values).T
    m, _ = np.shape(x)
    weights = np.asmatrix(np.eye(m, m))
    # 计算权重矩阵
    for i in range(m):
        difference_matrix = x[i, :] - test_point
        weights[i, i] = np.exp(
            (difference_matrix * difference_matrix.T).item() / (-2 * k**2)
        )
    # 判断逆矩阵的存在性
    x_t_x = x.T * weights * x
    if np.linalg.det(x_t_x) == 0:
        print("This matrix is singular, cannot do inverset.")
    # 计算回归系数
    w = x_t_x.I * x.T * weights * y
    # 计算预测值
    predicted_value = test_point * w
    print(predicted_value)
    return predicted_value


def ridge_liner_regression(x, y, lam=0.2):
    x_t_x = x.T * x
    demon = x_t_x + np.asmatrix(np.eye(np.shape(x)[1])) * lam
    if np.linalg.det(demon) == 0:
        print("This matrix is singular, cannot do inverset.")
    w = demon.I * x.T * y
    return w


def ridge_liner_regression_test(data_set, real_values):
    x = np.asmatrix(data_set)
    y = np.asmatrix(real_values).T
    # 对数据进行标准化处理
    y_means = np.mean(y, 0)
    y_stands = y - y_means
    x_means = np.mean(x, 0)
    x_var = np.var(x, 0)
    x_stands = (x - x_means) / x_var
    number_of_test = 30
    w_matrix = np.asmatrix(np.zeros((number_of_test, np.shape(x)[1])))
    for i in range(number_of_test):
        w = ridge_liner_regression(x_stands, y_stands, np.exp(i - 10))
        w_matrix[i, :] = w.T
    return w_matrix


def stage_wise(data_set, real_values, eps=0.01, number_of_iter=100):
    x = np.asmatrix(data_set)
    y = np.asmatrix(real_values).T
    x_means = np.mean(x, 0)
    x_var = np.var(x, 0)
    x_stands = (x - x_means) / x_var
    y_means = np.mean(y, 0)
    y_stands = y - y_means
    m, n = np.shape(x)
    w_matrix = np.asmatrix(np.zeros((number_of_iter, n)))
    w = np.asmatrix(np.zeros((n, 1)))
    w_test = w.copy()
    w_max = w.copy()
    for i in range(number_of_iter):
        # 因为是贪心算法，所以我们会寻找最优特征的最优更新
        lowest_error = np.inf
        print(w.T)
        for j in range(n):
            for sign in [1, -1]:
                w_test = w.copy()
                w_test[j] += sign * eps
                y_hat = x_stands * w_test
                error_of_rss = rss_error(y_stands.A, y_hat.A)
                if error_of_rss < lowest_error:
                    w_max = w_test.copy()
                    lowest_error = error_of_rss
        w = w_max.copy()
        w_matrix[i, :] = w.T
    return w_matrix


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.asmatrix(xArr)
    yMat = np.asmatrix(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean  # can also regularize ys but will get smaller coef
    x_var = np.var(xMat, 0)
    xMat = (xMat - np.mean(xMat, 0)) / x_var
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))  # testing code remove
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rss_error(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


def rss_error(y, y_hat):
    return ((y - y_hat) ** 2).sum()


def test_locally_weighted_liner_regression(
    data_to_predicted, data_set, real_values, k=1.0
):
    test_point = np.asmatrix(data_to_predicted)
    print(test_point)
    m, _ = np.shape(test_point)
    y_hat = np.zeros((m, 1))
    for i in range(m):
        y_hat[i, :] = locally_weighted_liner_regression(
            test_point[i, :], data_set, real_values, k
        )
    return y_hat


if __name__ == "__main__":
    # data_set, real_values = load_data_set("./ex0.txt")
    # w = stand_liner_regression(data_set, real_values)
    # print(f"Weights: {w}")
    # x = np.asmatrix(data_set)
    # y = np.asmatrix(real_values)
    # y_hat = x * w
    # correlation_coefficient = np.corrcoef(y_hat.T, y)
    # print(f"Correlation Coefficient:\n {correlation_coefficient}")
    # y_hat = test_locally_weighted_liner_regression(
    #     data_set[0], data_set, real_values, 0.001
    # )
    # print(f"The real value is {real_values[0]}, The predicted value is {y_hat}.")
    data_set, real_values = load_data_set("./abalone.txt")
    # w_matrix = ridge_liner_regression_test(data_set, real_values)
    w_matrix = stage_wise(data_set, real_values, 0.01, 200)
    # w_matrix = stageWise(data_set, real_values, 0.01, 200)
    print(w_matrix)
