import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def better_scot_grand_ascent(data_set, label_vector, iter_num=150):
    data_set = np.array(data_set)
    label_vector = np.array(label_vector)
    m, n = data_set.shape
    weights = np.ones(n)
    for i in range(iter_num):
        data_index = list(range(m))
        for j in range(m):
            alpha = 4 / (1 + i + j) + 0.01
            rand_index = random.randint(0, len(data_index) - 1)
            h = sigmoid(sum(data_set[rand_index] * weights))
            error = label_vector[rand_index] - h
            weights = weights + error * alpha * data_set[rand_index]
            del data_index[rand_index]
    return weights


def classify(data_vector, weights):
    data_vector = np.array(data_vector)
    weights = np.array(weights)
    prob = sigmoid(np.sum(data_vector * weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def logistic_test_death_rate():
    training_data_set = []
    training_label_vector = []
    with open("./horseColicTraining.txt") as fr_training:
        for line in fr_training.readlines():
            row_data_vector = line.strip().split("\t")
            data_vector = [float(row_data_vector[i]) for i in range(21)]
            training_data_set.append(data_vector)
            training_label_vector.append(float(row_data_vector[21]))
    weights = better_scot_grand_ascent(
        training_data_set, training_label_vector, iter_num=1000
    )
    # print(f"Weights: {weights}")
    error_count = 0
    number_of_test = 0
    with open("./horseColicTest.txt") as fr_testing:
        for line in fr_testing.readlines():
            row_data_vector = line.strip().split("\t")
            data_vector = [float(row_data_vector[i]) for i in range(21)]
            # print(data_vector)
            predicted_value = classify(data_vector, weights)
            real_value = float(row_data_vector[21])
            if predicted_value != real_value:
                error_count += 1
            number_of_test += 1
            print(
                f"The value of predicted is {predicted_value}; The real value is {real_value}."
            )
    error_rate = error_count / number_of_test
    print(f"The error rate of test is {error_rate * 100}%")
    return error_rate


def multi_test_of_death_rate(number_of_test=50):
    error_rate = 0
    for _ in range(number_of_test):
        error_rate += logistic_test_death_rate()
    print(f"The average error rate is {error_rate / number_of_test * 100}")


def load_data_set():
    data_matrix = []
    label_vector = []
    with open("./testSet.txt") as fr:
        for line in fr.readlines():
            data = line.strip().split("\t")
            data_matrix.append([1.0, float(data[0]), float(data[1])])
            label_vector.append(float(data[2]))
    return data_matrix, label_vector


if __name__ == "__main__":
    # data_set, label_vector = load_data_set()
    # weights = better_scot_grad_ascent(data_set, label_vector)
    # print(weights)
    # logistic_test_death_rate()
    multi_test_of_death_rate()
