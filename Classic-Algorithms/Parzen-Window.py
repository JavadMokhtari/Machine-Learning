import numpy as np
from math import sqrt

RADIUS = 2


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = sqrt(distance)
    return distance


class ParzenWindow:

    def __init__(self, radius):
        self.radius = radius

    # Is defined for calculating Euclidean distance between two row of data

    # Returns the K nearest neighbors of test_row as a list
    def get_window(self, train, test_row):
        distances = list()
        for train_row in train:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        window = list()
        for tup in distances:
            if tup[1] <= self.radius:
                window.append(tup[0])
        return window

    # We can classify an input data by the trining data with predict_classification() function
    def predict_classification(self, train, test_row):
        window = self.get_window(train, test_row)
        output_values = [row[-1] for row in window]
        if output_values:
            prediction = max(set(output_values), key=output_values.count)
            return prediction
        else:
            return

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
    parzen_algorithm = ParzenWindow(RADIUS)
    # With this loop, we classify all the test dataset row and measure performance
    for i in range(len(data_test_out)):
        prediction = parzen_algorithm.predict_classification(train_data, data_test_in[i])
        real_value = data_test_out[i]
        print('For {} Expected {}, Got {}.'.format(data_test_in[i], real_value, prediction), end='  ')
        if real_value == prediction:
            correct += 1
            print('True')
        else:
            print('False')
    performance = correct / len(data_test_out)
    print("\nAccuracy on test data: {:.2f}%\n".format(performance * 100))
    radius_values = np.arange(0.1, 10, 0.05)
    accuracies = list()
    for radius in radius_values:
        parzen_algorithm = ParzenWindow(radius)
        accuracy = parzen_algorithm.calculate_accuracy(train_data, data_test_in, data_test_out)
        accuracies.append((radius, accuracy))
    accuracies.sort(key=lambda tup: (-tup[1], tup[0]))
    print("The best case for test data is:\nRadius = {:.2f}\nAccuracy = {:.2f}%".format(accuracies[0][0],
                                                                                        accuracies[0][1] * 100))


if __name__ == "__main__":
    main()
