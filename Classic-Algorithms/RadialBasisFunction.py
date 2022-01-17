import numpy as np
import importlib
from math import sqrt
FindOptimal_K = importlib.import_module('FindOptimal-K-means', '*')


# Is used as kernel function in input layer of RBF
def Gaussian(r, var=1):
    return np.exp(- (r ** 2) / (2 * var))


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Is defined for calculating Euclidean distance between two rows of data
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = sqrt(distance)
    return distance


class RadialBasisFunction:
    # Initialization function
    def __init__(self, dataset, num_input=4, num_output=3, learning_rate=0.01, epoch=2000):
        self.dataset = dataset
        self.num_input = num_input
        self.num_output = num_output

        find_optimal_K = FindOptimal_K.__getattribute__('find_optimal_K')
        self.num_hidden = find_optimal_K(self.dataset)
        print("\nThe optimal number of hidden neurons was found:\nK = {}\n".format(self.num_hidden))

        self.learning_rate = learning_rate
        self.epoch = epoch
        self.output_layer_weights = np.random.uniform(-2, 2, (self.num_hidden, self.num_output))
        self.output_layer_biases = np.random.uniform(-2, 2, (1, self.num_output))
        self.centers = self.K_means_centers(self.dataset[:, :-1])
        self.hidden_neurons = np.empty((1, self.num_hidden))

    def K_means_centers(self, dataset):
        dataset_size = dataset.shape[0]
        indx = np.random.choice(dataset_size, self.num_hidden, replace=False)
        centers = dataset[indx, :]
        new_centers = np.empty(centers.shape)
        cluster_values = np.empty((dataset_size, 1))
        error_vector = np.empty(self.num_hidden)
        error = 1.000
        iteration = 0

        while error > 0.005:
            for i in range(dataset_size):
                min_dist = float('inf')
                for j in range(self.num_hidden):
                    dist = euclidean_distance(dataset[i], centers[j, :])
                    if dist <= min_dist:
                        min_dist = dist
                        cluster_values[i, :] = j

            # Set a cluster to each dataset row
            clustered_dataset = np.column_stack((dataset, cluster_values))
            # Updating centers
            for i in range(self.num_hidden):
                cluster = clustered_dataset[np.where(clustered_dataset[:, -1] == i)]

                if cluster.shape[0] == 0:
                    indx = np.random.randint(dataset_size)
                    new_center = dataset[indx, :]
                else:
                    column_values = [cluster[:, i] for i in range(cluster.shape[1] - 1)]
                    new_center = np.array([np.sum(column_values[i]) / cluster.shape[0]
                                           for i in range(cluster.shape[1] - 1)])
                new_centers[i, :] = new_center

            # Calculating distance between new centers and previous centers
            for i in range(self.num_hidden):
                error_vector[i] = euclidean_distance(new_centers[i, :], centers[i, :])
            error = sqrt(np.dot(error_vector, error_vector))

            iteration += 1
            centers = np.copy(new_centers)
        return centers

    def train(self, dataset):
        for epoch in range(self.epoch):
            for row in dataset:
                for i in range(self.num_hidden):
                    r = euclidean_distance(row[:-1], self.centers[i, :])
                    self.hidden_neurons[0][i] = Gaussian(r)

                # Calculate error in output layer for use in delta rule
                output = sigmoid(np.dot(self.hidden_neurons, self.output_layer_weights) + self.output_layer_biases)
                desired_output = np.zeros((1, self.num_output))
                desired_output[0][int(row[-1])] = 1
                error = np.subtract(desired_output, output)

                # Updating output layer weights
                self.output_layer_weights += self.learning_rate * np.dot(self.hidden_neurons.T, error)
                self.output_layer_biases += self.learning_rate * error

    def predict_classification(self, data_row):
        for i in range(self.num_hidden):
            r = euclidean_distance(data_row, self.centers[i, :])
            self.hidden_neurons[0][i] = Gaussian(r)
        output = sigmoid(np.dot(self.hidden_neurons, self.output_layer_weights) + self.output_layer_biases)
        predict = list(map(round, output[0]))
        for p in predict:
            if p == 1:
                return predict.index(p)

    # Make prediction and measure algorithm performance
    def calculate_performance(self, data_in, data_out):
        correct = 0
        for i in range(data_out.shape[0]):
            prediction = self.predict_classification(data_in[i])
            real_value = data_out[i]
            if real_value == prediction:
                correct += 1
        accuracy = correct / data_out.shape[0]
        return accuracy

    def K_fold_validation(self, train_data, k=5):
        np.random.shuffle(train_data)
        validation_data = np.split(train_data, k)
        accuracy_list = list()

        for i in range(k):
            index = [j * (i+1) for j in range(train_data.shape[0] // k)]
            train_data_clone = np.delete(train_data, index, 0)
            self.train(train_data_clone)
            accuracy = self.calculate_performance(validation_data[i][:, :-1], validation_data[i][:, -1])
            accuracy_list.append(accuracy)

        performance = sum(accuracy_list)*100/k
        return performance


def main():
    # Import train data
    input_data = np.loadtxt('../Data/iris/iris_train.csv', delimiter=',')
    output_data = np.loadtxt('../Data/iris/iris_train_label.csv', delimiter=',')
    train_data = np.column_stack((input_data, output_data))

    data_test_in = np.loadtxt('../Data/iris/iris_test.csv', delimiter=',')
    data_test_out = np.loadtxt('../Data/iris/iris_test_label.csv', delimiter=',')

    rbf = RadialBasisFunction(train_data)
    rbf.train(train_data)

    correct = 0
    for i in range(data_test_out.shape[0]):
        prediction = rbf.predict_classification(data_test_in[i])
        real_value = data_test_out[i]
        print('For {} Expected {}, Got {}.'.format(data_test_in[i], real_value, prediction), end='  ')
        if real_value == prediction:
            correct += 1
            print('True')
        else:
            print('False')
    performance = correct / data_test_out.shape[0]
    print("\nAccuracy on test dataset: {:.2f}%".format(performance * 100))
    print('\nPlease wait ...\n')
    print('Performance with K fold validation:\n{:.2f}%'.format(rbf.K_fold_validation(train_data)))


if __name__ == "__main__":
    main()
