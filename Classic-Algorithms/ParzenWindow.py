import numpy as np
from os import path


# Is defined for calculating Euclidean distance between two row of data
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = np.sqrt(distance)
    return distance


class ParzenWindow:
    def __init__(self, train_data, radius=1):
        self.radius = radius
        self.train_data = train_data

    # We can classify an input data by the training data with predict_classification() function
    def predict_classification(self, test_row):
        distances = list()
        for train_row in self.train_data:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))

        window = list()
        for tup in distances:
            if tup[1] <= self.radius:
                window.append(tup[0])

        output_values = [row[-1] for row in window]
        if output_values:
            prediction = max(set(output_values), key=output_values.count)
            return prediction
        else:
            return 0

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
    parzen_algorithm = ParzenWindow(train_data)
    # With this loop, we classify all the test dataset row and measure performance
    for i in range(len(data_test_out)):
        prediction = parzen_algorithm.predict_classification(data_test_in[i])
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
        parzen_algorithm = ParzenWindow(train_data, radius)
        accuracy = parzen_algorithm.calculate_accuracy(data_test_in, data_test_out)
        accuracies.append((radius, accuracy))
    accuracies.sort(key=lambda tup: (-tup[1], tup[0]))
    print("The best case for test data is:\nRadius = {:.2f}\nAccuracy = {:.2f}%"
          .format(accuracies[0][0], accuracies[0][1] * 100))


if __name__ == "__main__":
    main()
