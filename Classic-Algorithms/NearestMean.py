import numpy as np


# Is defined for calculating Euclidean distance between two row of data
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = np.sqrt(distance)
    return distance


# We can classify an input data by the training data with predict_classification() function
def predict_classification(train, test_row):
    output_train = set(train[:, -1])
    distances = list()
    for output in output_train:
        class_data = train[np.where(train[:, -1] == output)]
        column_values = [class_data[:, i] for i in range(class_data.shape[1] - 1)]
        average_row = [np.sum(column_values[i]) / class_data.shape[0] for i in range(class_data.shape[1] - 1)]
        dist = euclidean_distance(test_row, average_row)
        distances.append((average_row + [output], dist))
    distances.sort(key=lambda tup: tup[1])
    prediction = distances[0][0][-1]
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
    # With this loop, we classify all the test dataset row and measure performance
    for i in range(len(data_test_out)):
        prediction = predict_classification(train_data, data_test_in[i])
        real_value = data_test_out[i]
        print('For {} Expected {}, Got {}.'.format(data_test_in[i], real_value, prediction), end='  ')
        if real_value == prediction:
            correct += 1
            print('True')
        else:
            print('False')
    performance = correct / len(data_test_out)
    print("\nAccuracy on test dataset: {:.2f}%\n".format(performance * 100))


if __name__ == "__main__":
    main()
