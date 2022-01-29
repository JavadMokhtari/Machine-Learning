import numpy as np
from Kmeans import Kmeans
from os import path


# Is used as kernel function in input layer of RBF
def Gaussian(r, var):
    if var is None:
        var = 1
    return np.exp(- (r ** 2) / (2 * var))


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Is defined for calculating Euclidean distance between two rows of data
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = np.sqrt(distance)
    return distance


class RBF:
    # Initialization function
    def __init__(self, dataset, num_input=4, num_output=3, learning_rate=0.01, epoch=1000):
        self.dataset = dataset
        self.num_input = num_input
        self.num_output = num_output

        kmeans = Kmeans()
        self.num_hidden = kmeans.find_optimal_K(self.dataset)
        # print("\nThe optimal number of hidden neurons was found:\nK = {}\n".format(self.num_hidden))

        self.learning_rate = learning_rate
        self.epoch = epoch
        self.output_layer_weights = np.random.uniform(-2, 2, (self.num_hidden, self.num_output))
        self.output_layer_biases = np.random.uniform(-2, 2, (1, self.num_output))
        self.hidden_neurons = np.empty((1, self.num_hidden))

        kmeans.K_means(self.dataset, self.num_hidden)
        self.centers = kmeans.centers
        self.spreads = kmeans.clusters_variance

    def train(self):
        for epoch in range(self.epoch):
            for row in self.dataset:
                for i in range(self.num_hidden):
                    r = euclidean_distance(row[:-1], self.centers[i, :])
                    sig = self.spreads[i][0]
                    self.hidden_neurons[0][i] = Gaussian(r, sig)

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
            sig = self.spreads[i][0]
            self.hidden_neurons[0][i] = Gaussian(r, sig)
        output = sigmoid(np.dot(self.hidden_neurons, self.output_layer_weights) + self.output_layer_biases)
        predict = list(map(round, output[0]))
        for p in predict:
            if p == 1:
                return predict.index(p)
        return 0

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
            self.train()
            accuracy = self.calculate_performance(validation_data[i][:, :-1], validation_data[i][:, -1])
            accuracy_list.append(accuracy)
            print("{:.2f}%".format(accuracy * 100))

        performance = sum(accuracy_list)*100/k
        return performance


def main():
    # Import train data
    input_path = path.join('..', 'Data', 'iris', 'iris_train.csv')
    input_data = np.loadtxt(input_path, delimiter=',')
    output_path = path.join('..', 'Data', 'iris', 'iris_train_label.csv')
    output_data = np.loadtxt(output_path, delimiter=',')
    train_data = np.column_stack((input_data, output_data))

    # Import test data
    test_input_path = path.join('..', 'Data', 'iris', 'iris_test.csv')
    input_test_data = np.loadtxt(test_input_path, delimiter=',')
    test_output_path = path.join('..', 'Data', 'iris', 'iris_test_label.csv')
    output_test_data = np.loadtxt(test_output_path, delimiter=',')

    rbf = RBF(train_data)
    rbf.train()

    correct = 0
    for i in range(output_test_data.shape[0]):
        prediction = rbf.predict_classification(input_test_data[i])
        real_value = output_test_data[i]
        print('For {} Expected {}, Got {}.'.format(input_test_data[i], real_value, prediction), end='  ')
        if real_value == prediction:
            correct += 1
            print('True')
        else:
            print('False')
    performance = correct / output_test_data.shape[0]
    print("\nAccuracy on test dataset: {:.2f}%".format(performance * 100))
    print('\nPerformance with K fold validation:\n')
    print('\nAverage = {:.2f}%'.format(rbf.K_fold_validation(train_data)))


if __name__ == "__main__":
    main()
