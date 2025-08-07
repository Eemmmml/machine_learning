import random
import numpy as np


def load_data_set(filename):
    data_set = []
    label_vector = []
    with open(filename) as fr:
        for line in fr.readlines():
            data_vector = line.strip().split("\t")
            data_set.append([float(data_vector[0]), float(data_vector[1])])
            label_vector.append(float(data_vector[2]))
    return data_set, label_vector


def select_j_rand(i, m):
    """
    i 为当前已选择的alpha的下标
    m 为所有alpha的数量
    """
    j = i
    while j == i:
        j = random.randint(0, m - 1)
    return j


def clip_alpha(alpha, upper_bound, lower_bound):
    """
    裁剪alpha保证alpha的值属于[lower_bound, upper_bound]
    """
    if alpha > upper_bound:
        alpha = upper_bound
    elif alpha < lower_bound:
        alpha = lower_bound
    return alpha


# def smo_simple(data_set, class_labels, c, toler, max_iter_nubmer):
#     data_matrix = np.asmatrix(data_set)
#     label_vector = np.asmatrix(class_labels).transpose()
#     m, _ = np.shape(data_matrix)
#     alphas = np.asmatrix(np.zeros((m, 1)))
#     b = 0
#     iter_number = 0
#     while iter_number < max_iter_nubmer:
#         alpha_pair_changed = 0
#         for i in range(m):
#             fxi = float(
#                 np.multiply(label_vector, alphas).transpose()
#                 * (data_matrix * data_matrix[i, :].transpose())
#                 + b
#             )
#             ei = fxi - label_vector[i]
#             # 判断当前选择的alpha_i是否违背KTT条件，如果违反意味着可以被优化
#             if (ei * label_vector[i] < -toler and alphas[i] < c) or (
#                 ei * label_vector[i] > toler and alphas[i] > 0
#             ):
#                 j = select_j_rand(i, m)
#                 fxj = float(
#                     np.multiply(label_vector, alphas).transpose()
#                     * (data_matrix * data_matrix[j, :].transpose())
#                     + b
#                 )
#                 ej = fxj - label_vector[j]
#                 alpha_i_old = alphas[i].copy()
#                 alpha_j_old = alphas[j].copy()
#                 # 计算alpha的上下限
#                 if label_vector[i] != label_vector[j]:
#                     upper_bound = min(c, c - alphas[i] + alphas[j])
#                     lower_bound = max(0, alphas[j] - alphas[i])
#                 else:
#                     upper_bound = min(c, alphas[i] + alphas[j])
#                     lower_bound = max(0, alphas[i] + alphas[j] - c)
#                 eta = (
#                     data_matrix[i, :] * data_matrix[i, :].transpose()
#                     - 2 * data_matrix[i, :] * data_matrix[j, :].transpose()
#                     + data_matrix[j, :] * data_matrix[j, :].transpose()
#                 )
#                 if upper_bound == lower_bound:
#                     print("upper_bound == lower_bound")
#                     continue
#                 # 判断eta的合法性
#                 if eta <= 0:
#                     print("eta <= 0")
#                     continue
#                 alphas[j] += label_vector[j] * (ei - ej) / eta
#                 # 截取alpha_j保证其在[lower_bound, upper_bound]
#                 alphas[j] = clip_alpha(alphas[j], upper_bound, lower_bound)
#                 if abs(alphas[j] - alpha_j_old) < 0.00001:
#                     print("j not moving enough!!")
#                     continue
#                 alphas[i] += (
#                     label_vector[i] * label_vector[j] * (alpha_j_old - alphas[j])
#                 )
#                 b1 = (
#                     b
#                     - ei
#                     - (alphas[i] - alpha_i_old)
#                     * label_vector[i]
#                     * (data_matrix[i, :] * data_matrix[i, :].transpose())
#                     - (alphas[j] - alpha_j_old)
#                     * label_vector[j]
#                     * (data_matrix[i, :] * data_matrix[j, :].transpose())
#                 )
#                 b2 = (
#                     b
#                     - ej
#                     - (alphas[i] - alpha_i_old)
#                     * label_vector[i]
#                     * (data_matrix[i, :] * data_matrix[j, :].transpose())
#                     - (alphas[j] - alpha_j_old)
#                     * label_vector[j]
#                     * (data_matrix[j, :] * data_matrix[j, :].transpose())
#                 )
#                 if 0 < alphas[i] < c and 0 < alphas[j] < c:
#                     b = (b1 + b2) / 2
#                 elif 0 < alphas[i] < c:
#                     b = b1
#                 elif 0 < alphas[j] < c:
#                     b = b2
#                 else:
#                     b = (b1 + b2) / 2.0
#                 alpha_pair_changed += 1
#                 print(
#                     f"iter: {iter_number}, i: {i}, alpha pair changed: {alpha_pair_changed}"
#                 )
#         if alpha_pair_changed == 0:
#             iter_number += 1
#         else:
#             iter_number = 0
#         print(f"iter number: {iter_number}")
#     return b, alphas


class OptStruct:
    def __init__(self, data_matrix, class_label, c, toler, k_tuple=("lin", 0)):
        self.data_matrix = data_matrix
        self.label_vector = class_label
        self.c = c
        self.toler = toler
        self.m = np.shape(data_matrix)[0]
        self.alphas = np.asmatrix(np.zeros((self.m, 1)))
        self.b = 0
        self.ecache = np.asmatrix(np.zeros((self.m, 2)))
        self.kernel_trans_results = np.asmatrix(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.kernel_trans_results[:, i] = kernel_trans(
                self.data_matrix, self.data_matrix[i, :], k_tuple
            )


def calculate_ek(opt_struct, k):
    fxk = float(
        np.multiply(opt_struct.alphas, opt_struct.label_vector).transpose()
        * opt_struct.kernel_trans_results[:, k]
        + opt_struct.b
    )
    ek = float(fxk - opt_struct.label_vector[k])
    return ek


def select_j(opt_struct, i, ei):
    max_k = -1
    max_delta_ej = 0
    ej = 0
    # e_cache[0]表明缓存有效性1有效0无效，e_cache[1]记录缓存值
    opt_struct.ecache[i] = np.array([1, ei])
    valid_ecache_list = np.nonzero(opt_struct.ecache[:, 0].A)[0]
    if len(valid_ecache_list) > 1:
        for k in valid_ecache_list:
            if k == i:
                continue
            ek = calculate_ek(opt_struct, k)
            delta_ej = abs(ei - ek)
            if delta_ej > max_delta_ej:
                max_k = k
                max_delta_ej = delta_ej
                ej = ek
        return max_k, ej
    else:
        j = select_j_rand(i, opt_struct.m)
        ej = calculate_ek(opt_struct, j)
        return j, ej


def update_ecache(opt_struct, k):
    ek = calculate_ek(opt_struct, k)
    opt_struct.ecache[k] = [1, ek]


def innerl(opt_struct, i):
    ei = calculate_ek(opt_struct, i)
    if (
        ei * opt_struct.label_vector[i] < -opt_struct.toler
        and opt_struct.alphas[i] < opt_struct.c
    ) or (
        ei * opt_struct.label_vector[i] > opt_struct.toler and opt_struct.alphas[i] > 0
    ):
        j, ej = select_j(opt_struct, i, ei)
        alpha_i_old = opt_struct.alphas[i].copy()
        alpha_j_old = opt_struct.alphas[j].copy()
        # 计算alpha_j的上下界，并判断合法性
        if opt_struct.label_vector[i] != opt_struct.label_vector[j]:
            upper_bound = min(opt_struct.c, opt_struct.c - alpha_i_old + alpha_j_old)
            lower_bound = max(0, alpha_j_old - alpha_i_old)
        else:
            upper_bound = min(opt_struct.c, alpha_i_old + alpha_j_old)
            lower_bound = max(0, alpha_i_old + alpha_j_old - opt_struct.c)
        if upper_bound == lower_bound:
            print("upper bound equals lower bound!!")
            return 0
        # 计算eta值并判断其合法性
        eta = (
            opt_struct.kernel_trans_results[i, i]
            - 2 * opt_struct.kernel_trans_results[i, j]
            + opt_struct.kernel_trans_results[j, j]
        )
        if eta <= 0:
            print("eta <= 0!!")
            return 0
        # 计算迭代后的新的alpha_j
        opt_struct.alphas[j] += opt_struct.label_vector[j] * (ei - ej) / eta
        opt_struct.alphas[j] = clip_alpha(
            opt_struct.alphas[j], upper_bound, lower_bound
        )
        if abs(opt_struct.alphas[j] - alpha_j_old) < 0.00001:
            print("alpha j not moving enough!!")
            return 0
        opt_struct.alphas[i] += (
            opt_struct.label_vector[i]
            * opt_struct.label_vector[j]
            * (alpha_j_old - opt_struct.alphas[j])
        )
        opt_struct.alphas[i] = clip_alpha(
            opt_struct.alphas[i], upper_bound, lower_bound
        )
        # 更新缓存
        update_ecache(opt_struct, i)
        update_ecache(opt_struct, j)
        # 计算迭代后的b值
        bi = (
            opt_struct.b
            - ei
            - (opt_struct.alphas[i] - alpha_i_old)
            * opt_struct.label_vector[i]
            * opt_struct.kernel_trans_results[i, i]
            - (opt_struct.alphas[j] - alpha_j_old)
            * opt_struct.label_vector[j]
            * opt_struct.kernel_trans_results[i, j]
        )
        bj = (
            opt_struct.b
            - ej
            - (opt_struct.alphas[i] - alpha_i_old)
            * opt_struct.label_vector[i]
            * opt_struct.kernel_trans_results[j, i]
            - (opt_struct.alphas[j] - alpha_j_old)
            * opt_struct.label_vector[j]
            * opt_struct.kernel_trans_results[j, j]
        )
        if (
            0 < opt_struct.alphas[i] < opt_struct.c
            and 0 < opt_struct.alphas[j] < opt_struct.c
        ):
            opt_struct.b = (bi + bj) / 2
        elif 0 < opt_struct.alphas[i] < opt_struct.c:
            opt_struct.b = bi
        elif 0 < opt_struct.alphas[j] < opt_struct.c:
            opt_struct.b = bj
        else:
            opt_struct.b = (bi + bj) / 2
        return 1
    else:
        return 0


def smo_p(data_set, class_label, c, toler, max_iter_number, k_tuple=("lin", 0)):
    opt_struct = OptStruct(
        np.asmatrix(data_set), np.asmatrix(class_label).transpose(), c, toler, k_tuple
    )
    iter_number = 0
    entire_set = True
    alpha_pairs_changed_number = 0
    while iter_number < max_iter_number and (
        alpha_pairs_changed_number > 0 or entire_set
    ):
        alpha_pairs_changed_number = 0
        # 对整个数据集进行遍历
        if entire_set:
            iter_number += 1
            for i in range(opt_struct.m):
                alpha_pairs_changed_number += innerl(opt_struct, i)
                print(
                    f"Full Set, Iter: {iter_number}, i: {i}, Alpha Pairs Changed: {alpha_pairs_changed_number}"
                )
        # 对非边界的（潜在支持向量进行遍历）
        else:
            nonbound_indexs = np.nonzero(
                (opt_struct.alphas.A > 0) * (opt_struct.alphas.A < opt_struct.c)
            )[0]
            iter_number += 1
            for i in nonbound_indexs:
                alpha_pairs_changed_number += innerl(opt_struct, i)
                print(
                    f"Non Bound Set, Iter: {iter_number}, i: {i}, Alpha Pairs Changed: {alpha_pairs_changed_number}"
                )
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed_number == 0:
            entire_set = True
        print("============================")
        print(f"Iter Number: {iter_number}")
        print("============================")
    return opt_struct.b, opt_struct.alphas


def calculate_w_vector(data_set, class_label, alphas_vector):
    data_matrix = np.asmatrix(data_set)
    label_vector = np.asmatrix(class_label).transpose()
    m, n = np.shape(data_matrix)
    w_vector = np.zeros((n, 1))
    for i in range(m):
        w_vector += np.multiply(alphas_vector[i] * label_vector[i], data_matrix[i, :].T)
    return w_vector


def kernel_trans(x1, x2, k_tuple):
    """
    x1, x2 均为矩阵(matrix)
    x1 可以是任意行数的矩阵
    x2 必须是一个一行的向量
    """
    m, n = np.shape(x1)
    kernel_trans_result = np.asmatrix(np.zeros((m, 1)))
    if k_tuple[0] == "lin":
        kernel_trans_result = x1 * x2.T
    elif k_tuple[0] == "rbf":
        for i in range(m):
            row_delta = x1[i, :] - x2
            kernel_trans_result[i] = row_delta * row_delta.T
        kernel_trans_result = np.exp(kernel_trans_result / (-1 * k_tuple[1] ** 2))
    return kernel_trans_result


def classify(x, w, b):
    """
    其中 w均为列向量
        x为数组
        b为数值
    """
    w = np.asmatrix(w)
    x = np.asmatrix(x).transpose()
    fx = w.T * x + b
    if fx >= 0:
        return 1
    else:
        return -1


def test_rbf(k1=1.3):
    data_set, class_label = load_data_set("./testSetRBF.txt")
    b, alphas = smo_p(data_set, class_label, 10, 0.0001, 10000, ("rbf", k1))
    sv_indexs = np.nonzero(alphas.A > 0)[0]
    data_matrix = np.asmatrix(data_set)
    label_vector = np.asmatrix(class_label).transpose()
    svs = data_matrix[sv_indexs]
    sv_label_vector = label_vector[sv_indexs]
    print(f"There are {len(sv_indexs)} support vectors")
    m, _ = np.shape(data_matrix)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(svs, data_matrix[i, :], ("rbf", k1))
        predicted_result = (
            kernel_eval.T * np.multiply(sv_label_vector, alphas[sv_indexs]) + b
        )
        if np.sign(predicted_result) != np.sign(label_vector[i]):
            error_count += 1
            print(
                f"The result of predicted is {np.sign(predicted_result)} but the real result is {np.sign(label_vector[i])}"
            )
    print(f"The error rate of training is {error_count / m * 100}%")

    data_set, class_label = load_data_set("./testSetRBF2.txt")
    data_matrix = np.asmatrix(data_set)
    label_vector = np.asmatrix(class_label).transpose()
    m, _ = np.shape(data_matrix)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(svs, data_matrix[i, :], ("rbf", k1))
        predicted_result = (
            kernel_eval.T * np.multiply(alphas[sv_indexs], sv_label_vector) + b
        )
        if np.sign(predicted_result) != np.sign(label_vector[i]):
            error_count += 1
            print(
                f"The result of predicted is {np.sign(predicted_result)} but the real result is {np.sign(label_vector[i])}"
            )
    print(f"The error rate of test is {error_count / m * 100}%")


if __name__ == "__main__":
    # filename = "./testSet.txt"
    # data_set, label_vector = load_data_set(filename)
    # b, alphas = smo_p(data_set, label_vector, 0.6, 0.001, 40)
    # w_vector = calculate_w_vector(data_set, label_vector, alphas)
    # print("===========================================================================")
    # print(b)
    # print("---------------------------------------------------------------------------")
    # print(alphas)
    # print("---------------------------------------------------------------------------")
    # print(w_vector)
    # print("-------------------------------TEST----------------------------------------")
    # x = np.asmatrix(data_set)[0]
    # class_of_x = classify(x, w_vector, b)
    # print(
    #     f"The predicted result of class of x is {class_of_x}; The real result of class of x is {label_vector[0]}"
    # )
    # print("=====================================================================")
    # print("=====================================================================")
    # print("=============================TEST1===================================")
    # print("=====================================================================")
    # print("=====================================================================")
    # test_rbf(k1=0.4)
    # print("=====================================================================")
    # print("=====================================================================")
    # print("=============================TEST2===================================")
    # print("=====================================================================")
    # print("=====================================================================")
    # test_rbf(k1=0.6)
    # print("=====================================================================")
    # print("=====================================================================")
    # print("=============================TEST3===================================")
    # print("=====================================================================")
    # print("=====================================================================")
    # test_rbf(k1=0.8)
    # print("=====================================================================")
    # print("=====================================================================")
    # print("=============================TEST4===================================")
    # print("=====================================================================")
    # print("=====================================================================")
    # test_rbf(k1=1.0)
    # print("=====================================================================")
    # print("=====================================================================")
    # print("=============================TEST5===================================")
    # print("=====================================================================")
    # print("=====================================================================")
    # test_rbf(k1=1.2)
    # print("=====================================================================")
    # print("=====================================================================")
    # print("=============================TEST6===================================")
    # print("=====================================================================")
    # print("=====================================================================")
    # test_rbf(k1=1.4)
    # print("=====================================================================")
    # print("=====================================================================")
    # print("=============================TEST7===================================")
    # print("=====================================================================")
    # print("=====================================================================")
    # test_rbf(k1=1.6)
    # print("=====================================================================")
    # print("=====================================================================")
    # print("=============================TEST8===================================")
    # print("=====================================================================")
    # print("=====================================================================")
    # test_rbf(k1=1.8)
    # print("=====================================================================")
    # print("=====================================================================")
    # print("=============================TEST9===================================")
    # print("=====================================================================")
    # print("=====================================================================")
    # test_rbf(k1=2.0)
    test_rbf(k1=1.3)
