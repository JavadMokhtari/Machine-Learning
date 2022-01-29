import numpy as np
from os import path

K = 10


# Is defined for calculating Euclidean distance between two rows of data
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = np.sqrt(distance)
    return distance


class KNN:
    def __init__(self, train_data, K):
        self.K = K
        self.neighbors = None
        self.train_data = train_data

    # We can classify an input data by the training data with predict_classification() function
    def predict_classification(self, test_row):
        # Getting the K nearest neighbors of test_row as a list
        distances = list()
        for row in self.train_data:
            dist = euclidean_distance(test_row, row)
            distances.append((row, dist))
        distances.sort(key=lambda tup: tup[1])

        # Adding all neighbors in a list
        neighbors = list()
        for i in range(self.K):
            neighbors.append(distances[i][0])

        # Return nearest neighbor label as input vector label
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction

    # We can calculate the algorithm accuracy with having real outputs
    def calculate_accuracy(self, test_in, test_out):
        correct = 0.0
        for i in range(test_in.shape[0]):
            prediction = self.predict_classification(test_in[i])
            real_value = test_out[i]
            if real_value == prediction:
                correct += 1.0
        accuracy = correct / test_in.shape[0]
        return accuracy


def main():
    # Import train data
    input_path = path.join('..', 'Data', 'iris', 'iris_train.csv')
    data_in = np.loadtxt(input_path, delimiter=',')
    output_path = path.join('..', 'Data', 'iris', 'iris_train_label.csv')
    data_out = np.loadtxt(output_path, delimiter=',')
    train_data = np.column_stack((data_in, data_out))

    # Import test data
    test_input_path = path.join('..', 'Data', 'iris', 'iris_test.csv')
    data_test_in = np.loadtxt(test_input_path, delimiter=',')
    test_output_path = path.join('..', 'Data', 'iris', 'iris_test_label.csv')
    data_test_out = np.loadtxt(test_output_path, delimiter=',')

    correct = 0
    knn_algorithm = KNN(train_data, K)
    # With this loop, we classify all the test dataset row and measure performance
    for i in range(len(data_test_out)):
        prediction = knn_algorithm.predict_classification(data_test_in[i])
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
