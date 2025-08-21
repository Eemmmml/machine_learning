import numpy as np


def load_data_set(filename):
    data_set = []
    with open(filename) as fr:
        for line in fr.readlines():
            data_in_line = line.strip().split("\t")
            data_set.append(list(map(float, data_in_line)))
    return data_set


def distance_euclidean(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1 - vector2, 2)))


def rand_centroids(data_matrix, k):
    _, n = np.shape(data_matrix)
    centroids = np.asmatrix(np.ones((k, n)))
    for feature in range(n):
        min_feature_value = np.min(data_matrix[:, feature])
        max_feature_value = np.max(data_matrix[:, feature])
        feature_value_range = max_feature_value - min_feature_value
        centroids[:, feature] = (
            min_feature_value + feature_value_range * np.random.rand(k, 1)
        )
    return centroids


def k_means(
    data_matrix, k, distance_type=distance_euclidean, create_centroids=rand_centroids
):
    m, _ = np.shape(data_matrix)
    centroids = create_centroids(data_matrix, k)
    cluster_changed = True
    cluster_assignment = np.zeros((m, 2))
    iter_number = 0
    while cluster_changed:
        iter_number += 1
        cluster_changed = False
        for i in range(m):
            min_distance = np.inf
            min_index = -1
            for j in range(k):
                distance = distance_type(centroids[j, :], data_matrix[i, :])
                if distance < min_distance:
                    min_distance = distance
                    min_index = j
            if cluster_assignment[i, 0] != min_index:
                cluster_changed = True
                cluster_assignment[i, :] = min_index, min_distance**2
        print(f"Centroids: {centroids}")
        for cent in range(k):
            point_in_cluster = data_matrix[
                np.nonzero(cluster_assignment[:, 0] == cent)[0]
            ]
            centroids[cent, :] = np.mean(point_in_cluster, axis=0)
    print(f"Iter Number: {iter_number}")
    return centroids, cluster_assignment


def binary_k_means(
    data_matrix, k, distance_type=distance_euclidean, create_centroids=rand_centroids
):
    m, _ = np.shape(data_matrix)
    # 初始将所用点视为在一个簇中
    centroid0 = np.mean(data_matrix, axis=0)
    # 初始化簇的质心列表
    centroids = [centroid0]
    # 初始化数据点簇分配序列
    cluster_assignment = np.zeros((m, 2))
    for i in range(m):
        # 初始化每个数据点和初始簇的距离平方
        cluster_assignment[i, 1] = (
            distance_type(np.asmatrix(centroid0), data_matrix[i, :]) ** 2
        )
    # 二元分割簇
    while len(centroids) < k:
        # 寻找分割后ssm下降最大的簇
        lowest_sse = np.inf
        best_cluster_assignment = None
        best_centroids = None
        best_cluster_to_split = -1
        for i in range(len(centroids)):
            point_in_cluster = data_matrix[
                np.nonzero(cluster_assignment[:, 0] == i)[0], :
            ]
            new_centroids, new_cluster_assignment = k_means(
                point_in_cluster, 2, distance_type, create_centroids
            )
            sse_of_split = np.sum(new_cluster_assignment[:, 1])
            sse_of_no_split = np.sum(
                cluster_assignment[np.nonzero(cluster_assignment[:, 0] != i)[0], 1]
            )
            if sse_of_no_split + sse_of_split < lowest_sse:
                best_cluster_to_split = i
                best_cluster_assignment = new_cluster_assignment.copy()
                best_centroids = new_centroids
                lowest_sse = sse_of_no_split + sse_of_split
        # 用找到的最优分割更新原本的数据
        if (
            (best_cluster_to_split != -1)
            and (best_cluster_assignment is not None)
            and (best_centroids is not None)
        ):
            best_cluster_assignment[
                np.nonzero(best_cluster_assignment[:, 0] == 1)[0], 0
            ] = len(centroids)
            best_cluster_assignment[
                np.nonzero(best_cluster_assignment[:, 0] == 0)[0], 0
            ] = best_cluster_to_split
            centroids[best_cluster_to_split] = best_centroids[0, :]
            centroids.append(best_centroids[1, :])
            cluster_assignment[
                np.nonzero(cluster_assignment[:, 0] == best_cluster_to_split)[0], :
            ] = best_cluster_assignment
        else:
            raise NameError("Cluster is None")
    centroids = np.asmatrix([centroid.A.tolist()[0] for centroid in centroids])
    return centroids, cluster_assignment


if __name__ == "__main__":
    # filename = "./testSet.txt"
    # data_matrix = np.asmatrix(load_data_set(filename))
    # centroids, cluster_assignment = k_means(data_matrix, 4)
    # print(centroids)
    filename = "./testSet2.txt"
    data_matrix = np.asmatrix(load_data_set(filename))
    centroids, cluster_assignment = binary_k_means(data_matrix, 3)
    print(f"Type: {type(centroids)}\n{centroids}")
