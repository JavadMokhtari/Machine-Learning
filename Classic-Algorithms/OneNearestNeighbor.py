import numpy as np
from os import path


# Is defined for calculating Euclidean distance between two row of data
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = np.sqrt(distance)
    return distance


# Returns the predicted value 
def predict_classification(train, test_row):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbor = list(distances[0][0])
    prediction = neighbor[-1]
    return prediction


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

    test_size = len(data_test_out)
    correct = 0
    # With this loop, we classify all the test dataset row and measure performance
    for i in range(test_size):
        prediction = predict_classification(train_data, data_test_in[i])
        real_value = data_test_out[i]
        print('For {} Expected {}, Got {}'.format(data_test_in[i], real_value, prediction), end='  ')
        if real_value == prediction:
            correct += 1
            print('True')
        else:
            print('False')
    performance = float(correct) / test_size
    print("\nAccuracy is:   {:.2f} %".format(performance * 100))


if __name__ == "__main__":
    main()
