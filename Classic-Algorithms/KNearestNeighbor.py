import numpy as np
from math import sqrt, log, ceil

K = 10


# Is defined for calculating Euclidean distance between two rows of data
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = sqrt(distance)
    return distance


class KNN:

    def __init__(self, K):
        self.K = K

    # Returns the K nearest neighbors of test_row as a list
    def get_neighbors(self, train, test_row):
        distances = list()
        for train_row in train:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.K):
            neighbors.append(distances[i][0])
        return neighbors

    # We can classify an input data by the training data with predict_classification() function
    def predict_classification(self, train, test_row):
        neighbors = self.get_neighbors(train, test_row)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction

    # We can calculate the algorithm accuracy with having real outputs
    def calculate_accuracy(self, train, test_in, test_out):
        correct = 0.0
        for i in range(test_in.shape[0]):
            prediction = self.predict_classification(train, test_in[i])
            real_value = test_out[i]
            if real_value == prediction:
                correct += 1.0
        accuracy = correct / test_in.shape[0]
        return accuracy


def main():
    # Import train data
    data_in = np.loadtxt('../Data/iris/iris_train.csv', delimiter=',')
    data_out = np.loadtxt('../Data/iris/iris_train_label.csv', delimiter=',')
    train_data = np.column_stack((data_in, data_out))

    # Import test data
    data_test_in = np.loadtxt('../Data/iris/iris_test.csv', delimiter=',')
    data_test_out = np.loadtxt('../Data/iris/iris_test_label.csv', delimiter=',')

    correct = 0
    knn_algorithm = KNN(K)
    # With this loop, we classify all the test dataset row and measure performance
    for i in range(len(data_test_out)):
        prediction = knn_algorithm.predict_classification(train_data, data_test_in[i])
        real_value = data_test_out[i]
        print('For {} Expected {}, Got {}.'.format(data_test_in[i], real_value, prediction), end='  ')
        if real_value == prediction:
            correct += 1
            print('True')
        else:
            print('False')
    performance = correct / len(data_test_out)
    print("\nAccuracy on test dataset: {}%".format(performance * 100))


if __name__ == "__main__":
    main()
