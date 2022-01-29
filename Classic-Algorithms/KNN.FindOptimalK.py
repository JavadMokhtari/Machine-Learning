import numpy as np
from os import path


# Is defined for calculating Euclidean distance between two row of data
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = np.sqrt(distance)
    return distance


# Returns the K nearest neighbors of test_row as a list
def get_neighbors(train, test_row, k_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# We can classify an input data by the training data with predict_classification() function
def predict_classification(train, test_row, k_neighbors):
    neighbors = get_neighbors(train, test_row, k_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# We can calculate the algorithm accuracy with having real outputs
def calculate_accuracy(train, test_in, test_out, k_neighbor):
    correct = 0.0
    for i in range(test_in.shape[0]):
        prediction = predict_classification(train, test_in[i], k_neighbor)
        real_value = test_out[i]
        if real_value == prediction:
            correct += 1.0
    accuracy = correct / test_in.shape[0]
    return accuracy


# Finding optimal K with validation technique
def find_optimal_K(train_data):
    results = list()
    for i in range(train_data.shape[0] // 10):
        train_data_clone = train_data
        data_size = train_data_clone.shape
        validation_size = data_size[0] // 10, data_size[1]
        validation_data = np.zeros(validation_size)
        K_range = np.ceil(np.log10(data_size[0])) * 10
        np.random.seed(1)
        for j in range(validation_size[0]):
            index = np.random.randint(data_size[0])
            validation_data[j, :] = train_data_clone[index]
            train_data_clone = np.delete(train_data_clone, index, 0)
            data_size = train_data_clone.shape
        k_values = [i for i in range(1, int(K_range))]
        for k in k_values:
            accuracy = calculate_accuracy(train_data_clone, validation_data[:, :-1], validation_data[:, -1], k)
            results.append((k, accuracy))
    results.sort(key=lambda tup: (-tup[1], tup[0]))
    best_k = results[0][0]
    return best_k


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

    k = find_optimal_K(train_data=train_data)
    correct = 0

    # With this loop, we classify all the test dataset row and measure performance
    for i in range(len(data_test_out)):
        prediction = predict_classification(train_data, data_test_in[i], k)
        real_value = data_test_out[i]
        print('For {} Expected {}, Got {}'.format(data_test_in[i], real_value, prediction), end='  ')
        if real_value == prediction:
            correct += 1
            print('True')
        else:
            print('False')
    performance = correct / len(data_test_out)
    print("\nOptimal K found: {}\n\nAccuracy on test data: {:.2f}%\n".format(k, performance * 100))
    k_values = [i for i in range(1, 30)]
    accuracies = list()
    for k in k_values:
        accuracies.append((k, calculate_accuracy(train_data, data_test_in, data_test_out, k)))
    accuracies.sort(key=lambda tup: (-tup[1], tup[0]))
    print("The best K for test data is: {}".format(accuracies[0][0]))


if __name__ == "__main__":
    main()
