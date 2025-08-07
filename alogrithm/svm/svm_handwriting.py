import numpy as np
import random


def load_data_set(filename):
    """
    从本地文件中加载数据集和标签列表
    """
    data_set = []
    class_labels = []
    with open(filename) as fr:
        for line in fr.readlines():
            data_vector = line.strip().split("\t")
            data_set.append([float(data_vector[0]), float(data_vector[1])])
            class_labels.append(float(data_vector[2]))
    return data_set, class_labels


def kernel_trans(matrix, vector, k_tuple=("lin", 0)):
    """
    计算核函数
    matrix 为一个矩阵
    vector 为一个行向量
    """
    m, _ = np.shape(matrix)
    kernel_trans_result = np.asmatrix(np.zeros((m, 1)))
    if k_tuple[0] == "lin":
        kernel_trans_result = matrix * vector.T
    elif k_tuple[0] == "rbf":
        for i in range(m):
            delta_vector = matrix[i, :] - vector
            kernel_trans_result[i] = delta_vector * delta_vector.T
        kernel_trans_result = np.exp(kernel_trans_result / (-1 * k_tuple[1] ** 2))
    return kernel_trans_result


class OptStruct:
    def __init__(self, data_set, class_label, c, toler, k_tuple=("lin", 0)):
        # 数据矩阵
        self.data_matrix = np.asmatrix(data_set)
        # 标签列向量
        self.label_vector = np.asmatrix(class_label).transpose()
        # 数据向量数
        self.m = np.shape(self.data_matrix)[0]
        # alphas 参数列向量
        self.alphas = np.asmatrix(np.zeros((self.m, 1)))
        # b 参数
        self.b = 0
        # 惩罚系数
        self.c = c
        # 软间隔的参数
        self.toler = toler
        # E_i的缓存
        self.ecache = np.zeros((self.m, 2))
        # 核函数计算结果矩阵
        self.kernel_trans_results = np.asmatrix(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.kernel_trans_results[:, i] = kernel_trans(
                self.data_matrix, self.data_matrix[i, :], k_tuple
            )


def calculate_ek(opt_struct: OptStruct, k):
    """
    计算E_k
    """
    fxk = float(
        (
            np.multiply(opt_struct.alphas, opt_struct.label_vector).transpose()
            * opt_struct.kernel_trans_results[:, k]
        ).item()
        + opt_struct.b
    )
    # print(f"Calculate fxk: {fxk}")
    ek = float(fxk - opt_struct.label_vector[k].item())
    # print(f"Calculate Ek: {ek}")
    return ek


def select_j(opt_struct: OptStruct, i, ei):
    """
    根据alpha_i选择alpha_j
    """
    max_j = -1
    max_delta_e = 0
    ej = 0
    # 首先从所以潜在的支持向量中寻找
    sv_indexs = [
        index
        for index in range(opt_struct.m)
        if 0 < opt_struct.alphas[index] < opt_struct.c
    ]
    if len(sv_indexs) > 0:
        for k in sv_indexs:
            if k == i:
                continue
            ek = calculate_ek(opt_struct, k)
            delta_e = abs(ei - ek)
            if delta_e > max_delta_e:
                max_j = k
                max_delta_e = delta_e
                ej = k
        if max_j != -1:
            return max_j, ej
    # 如果支持向量中没有则在全部数据中寻找
    for k in range(opt_struct.m):
        if k == i:
            continue
        ek = calculate_ek(opt_struct, k)
        delta_e = abs(ei - ek)
        if delta_e > max_delta_e:
            max_j = k
            max_delta_e = delta_e
            ej = k
    # 在全部数据中仍然没有选出，则进行随机选择
    if max_j == -1:
        j = i
        while j == i:
            j = random.randint(0, opt_struct.m - 1)
        ej = calculate_ek(opt_struct, j)
        return j, ej

    return max_j, ej


def clip_alhpa(alpha, lower_bound, upper_bound):
    """
    截取 alpha 保证其在区间 [lower_bound, upper_bound]
    """
    if alpha > upper_bound:
        alpha = upper_bound
    elif alpha < lower_bound:
        alpha = lower_bound
    return alpha


def update_ecache(opt_struct: OptStruct, k):
    """
    更新ek的缓存值
    """
    ek = calculate_ek(opt_struct, k)
    # print(ek)
    opt_struct.ecache[k] = np.array([ek, 1])


def innerl(opt_struct: OptStruct, i):
    """
    SMO函数的内循环
    """
    ei = calculate_ek(opt_struct, i)
    if (
        opt_struct.alphas[i] < opt_struct.c
        and opt_struct.label_vector[i].item() * ei < -opt_struct.toler
    ) or (
        opt_struct.alphas[i] > 0
        and opt_struct.label_vector[i].item() * ei > opt_struct.toler
    ):
        j, ej = select_j(opt_struct, i, ei)
        # 首先拷贝alpha_i alpha_j 迭代前的值，以便后续使用
        alpha_i_old = opt_struct.alphas[i].copy()
        alpha_j_old = opt_struct.alphas[j].copy()
        # 计算alpha_i 和 alpha_j 的上下界
        # 首先计算 alpha_i 和 alpha_j 对应的 y 值异号的情况
        if opt_struct.label_vector[i] != opt_struct.label_vector[j]:
            upper_bound = min(opt_struct.c, opt_struct.c - alpha_i_old + alpha_j_old)
            lower_bound = max(0, alpha_j_old - alpha_i_old)
        else:
            upper_bound = min(opt_struct.c, alpha_i_old + alpha_j_old)
            lower_bound = max(0, alpha_i_old + alpha_j_old - opt_struct.c)
        if upper_bound == lower_bound:
            print("upper bound equals lower bound!!")
            return 0
        # 计算 eta 值
        eta = (
            opt_struct.kernel_trans_results[i, i]
            - 2 * opt_struct.kernel_trans_results[i, j]
            + opt_struct.kernel_trans_results[j, j]
        )
        if eta <= 0:
            print("The value of eta lower than 0!!")
            return 0
        opt_struct.alphas[j] += opt_struct.label_vector[j] * (ei - ej) / eta
        opt_struct.alphas[j] = clip_alhpa(
            opt_struct.alphas[j], lower_bound, upper_bound
        )
        if abs(opt_struct.alphas[j] - alpha_j_old) < 0.00001:
            print("The alpha j not moving enough!!")
            return 0
        opt_struct.alphas[i] += (
            opt_struct.label_vector[i]
            * opt_struct.label_vector[j]
            * (alpha_j_old - opt_struct.alphas[j])
        )
        opt_struct.alphas[i] = clip_alhpa(
            opt_struct.alphas[i], lower_bound, upper_bound
        )
        # 计算迭代后的b的值
        b1 = (
            opt_struct.b
            - ei
            - opt_struct.label_vector[i]
            * (opt_struct.alphas[i] - alpha_i_old)
            * opt_struct.kernel_trans_results[i, i]
            - opt_struct.label_vector[j]
            * (opt_struct.alphas[j] - alpha_j_old)
            * opt_struct.kernel_trans_results[i, j]
        )
        b2 = (
            opt_struct.b
            - ej
            - opt_struct.label_vector[i]
            * (opt_struct.alphas[i] - alpha_i_old)
            * opt_struct.kernel_trans_results[i, j]
            - opt_struct.label_vector[j]
            * (opt_struct.alphas[j] - alpha_j_old)
            * opt_struct.kernel_trans_results[j, j]
        )
        # 计算实际的b值
        if (0 < opt_struct.alphas[i] < opt_struct.c) and (
            0 < opt_struct.alphas[j] < opt_struct.c
        ):
            opt_struct.b = (b1 + b2) / 2
        elif 0 < opt_struct.alphas[i] < opt_struct.c:
            opt_struct.b = b1
        elif 0 < opt_struct.alphas[j] < opt_struct.c:
            opt_struct.b = b2
        else:
            opt_struct.b = (b1 + b2) / 2
        update_ecache(opt_struct, i)
        update_ecache(opt_struct, j)
        return 1
    else:
        return 0


def smo(data_set, class_label, c, toler, max_iter_number=10000, k_tuple=("lin", 0)):
    """
    利用smo训练alphas, b 参数
    """
    opt_struct = OptStruct(data_set, class_label, c, toler, k_tuple)
    iter_number = 0
    entire_set = True
    alphas_changed_number = 0
    while iter_number < max_iter_number and (alphas_changed_number > 0 or entire_set):
        alphas_changed_number = 0
        # 遍历整个数据集的数据向量
        if entire_set:
            iter_number += 1
            for i in range(opt_struct.m):
                alphas_changed_number += innerl(opt_struct, i)
                print(
                    f"Full Set: Iter Number: {iter_number}; i: {i}, Alpha Changed Number: {alphas_changed_number}"
                )
        # 重点遍历潜在的支持向量
        else:
            iter_number += 1
            psv_indexs = np.nonzero(
                (opt_struct.alphas.A > 0) * (opt_struct.alphas.A < opt_struct.c)
            )[0]
            for i in psv_indexs:
                alphas_changed_number += innerl(opt_struct, i)
                print(
                    f"Non Bound Set: Iter Number: {iter_number}; i: {i}, Alpha Changed Number: {alphas_changed_number}"
                )
        if entire_set:
            entire_set = False
        elif alphas_changed_number == 0:
            entire_set = True
    print("=================================================================")
    print(
        f"==========================Iter Number: {iter_number}======================="
    )
    print("=================================================================")
    return opt_struct.b, opt_struct.alphas


def test_rbf(k1=1.3):
    data_set, class_label = load_data_set("./testSetRBF.txt")
    b, alphas = smo(data_set, class_label, 200, 0.0001, k_tuple=("rbf", k1))
    data_matrix = np.asmatrix(data_set)
    label_vector = np.asmatrix(class_label).transpose()
    sv_alphas_indexs = np.nonzero(alphas.A > 0)[0]
    svs = data_matrix[sv_alphas_indexs]
    sv_label_vector = label_vector[sv_alphas_indexs]
    print(f"There are {len(sv_alphas_indexs)} support vectors")
    m, _ = np.shape(data_matrix)
    # 训练集验证
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(svs, data_matrix[i, :], ("rbf", k1))
        predicted = (
            np.multiply(sv_label_vector, alphas[sv_alphas_indexs]).transpose()
            * kernel_eval
            + b
        )
        if np.sign(predicted) != np.sign(label_vector[i]):
            error_count += 1
    print(f"Training Set: Error Rate: {error_count / m * 100}%")
    # 测试集验证
    data_set, class_label = load_data_set("./testSetRBF2.txt")
    data_matrix = np.asmatrix(data_set)
    label_vector = np.asmatrix(class_label).transpose()
    m, _ = np.shape(data_matrix)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(svs, data_matrix[i, :], ("rbf", k1))
        predicted = (
            np.multiply(sv_label_vector, alphas[sv_alphas_indexs]).transpose()
            * kernel_eval
            + b
        )
        if np.sign(predicted) != np.sign(label_vector[i]):
            error_count += 1
    print(f"Training Set: Error Rate: {error_count / m * 100}%")
    # print(alphas[alphas > 0])
    # print(b)


def image2vector(filename):
    image_data_vector = np.empty((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            line = fr.readline().strip()
            for j in range(32):
                image_data_vector[0, i * 32 + j] = int(line[j])
    return image_data_vector


def load_images(dirname):
    from os import listdir

    handwriting_labels = []
    training_file_list = listdir(dirname)
    m = len(training_file_list)
    handwriting_data_set = np.empty((m, 1024))
    for i in range(m):
        file = training_file_list[i]
        filename = file.split(".")[0]
        number_of_file = int(filename.split("_")[0])
        if number_of_file == 9:
            handwriting_labels.append(1)
        else:
            handwriting_labels.append(-1)
        handwriting_data_set[i, :] = image2vector(f"{dirname}/{file}")
    return handwriting_data_set, handwriting_labels


def test_handwriting_rbf(k_tuple=("rbf", 10)):
    handwriting_data_set, handwriting_labels = load_images("./digits/trainingDigits/")
    b, alphas = smo(
        handwriting_data_set, handwriting_labels, 200, 0.0001, 10000, k_tuple
    )
    handwriting_data_matrix = np.asmatrix(handwriting_data_set)
    handwriting_label_vector = np.asmatrix(handwriting_labels).transpose()
    sv_alphas_indexs = np.nonzero(alphas > 0)[0]
    handwriting_svs = handwriting_data_matrix[sv_alphas_indexs]
    handwriting_sv_label_vector = handwriting_label_vector[sv_alphas_indexs]
    print(f"There are {len(sv_alphas_indexs)} Support Vectors")
    m, _ = np.shape(handwriting_data_matrix)
    error_count = 0
    # 进行训练集测试
    for i in range(m):
        kernel_eval = kernel_trans(
            handwriting_svs, handwriting_data_matrix[i, :], k_tuple
        )
        predicted = (
            kernel_eval.T
            * np.multiply(alphas[sv_alphas_indexs], handwriting_sv_label_vector)
            + b
        )
        if np.sign(predicted) != np.sign(handwriting_label_vector[i]):
            error_count += 1
    print(f"Training Data Set: The error rate is {error_count / m * 100}%")
    # 进行测试集测试
    handwriting_data_set, handwriting_labels = load_images("./digits/testDigits/")
    handwriting_data_matrix = np.asmatrix(handwriting_data_set)
    handwriting_label_vector = np.asmatrix(handwriting_labels).transpose()
    m, _ = np.shape(handwriting_data_matrix)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(
            handwriting_svs, handwriting_data_matrix[i, :], k_tuple
        )
        predicted = (
            kernel_eval.T
            * np.multiply(alphas[sv_alphas_indexs], handwriting_sv_label_vector)
            + b
        )
        if np.sign(predicted) != np.sign(handwriting_label_vector[i]):
            error_count += 1
    print(f"Test Data Set: The error rate is {error_count / m * 100}%")


if __name__ == "__main__":
    # test_rbf()
    test_handwriting_rbf(k_tuple=("rbf", 25.5))
