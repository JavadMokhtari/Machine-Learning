import numpy as np
from os import path


class MLP:

    def __init__(self, num_input=4, hidden_layers=None, num_output=3, learning_rate=0.01, epoch=1000):

        # hidden_layers is a list that contains number of neurons in each layer
        if hidden_layers is None:
            hidden_layers = [5]

        self.num_input = num_input
        self.num_hidden = hidden_layers
        self.num_output = num_output
        self.learning_rate = learning_rate
        self.epoch = epoch

        # Set weights and biases for input data, hidden layers
        layers = [num_input] + hidden_layers + [num_output]
        weights = list()
        biases = list()
        for i in range(len(layers) - 1):
            # np.random.seed(1)
            weights.append(np.random.normal(0, 1, (layers[i], layers[i + 1])))
            biases.append(np.random.normal(0, 1, (1, layers[i + 1])))
        self.weights = weights
        self.biases = biases

    # Activation function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward(self, net_input_row):
        net_neurons = list()

        net_input_row = np.reshape(net_input_row, (1, self.num_input))
        net_neurons.append((net_input_row, 0))

        # loop for computing net nodes value in matrix
        for i in range(len(self.weights)):
            net_input_row = self.sigmoid(np.dot(net_input_row, self.weights[i]) + self.biases[i])
            net_neurons.append((net_input_row, i + 1))

        return net_neurons

    def backpropagation(self, net_input_row, class_label):
        # convert desired class value to numpy array
        desired_output = np.zeros((1, self.num_output))
        desired_output[0][int(class_label)] = 1

        # The obtained output from forward
        net_neurons = self.forward(net_input_row)
        output = net_neurons[-1][0]

        # find the errors for the output layer,
        # also update the weights between hidden layer and output layer
        delta = np.empty(output.shape)
        for i in range(output.shape[1]):
            delta[0][i] = np.subtract(desired_output[0][i], output[0][i]) * output[0][i] * (1 - output[0][i])

        # (n-1)th layer input
        mid_input = np.transpose(net_neurons[-2][0])

        # computing delta weights and updating weights for output layer
        delta_weights = self.learning_rate * np.dot(mid_input, delta)
        self.weights[-1] += delta_weights

        # updating biases for output layer
        delta_biases = self.learning_rate * delta
        self.biases[-1] += delta_biases

        # Updating weights for other layers
        for layer in range(-2, -len(self.weights) - 1, -1):
            # create an array for appending nodes delta values
            delta_hid = np.empty(net_neurons[layer][0].shape)

            for j in range(net_neurons[layer][0].shape[1]):
                # jth node value in lth layer
                mid_output = net_neurons[layer][0][0][j]
                delta_hid[0][j] = mid_output * (1 - mid_output) * np.dot(self.weights[layer+1], delta.T)[j][0]

            # (l-1)th layer input
            mid_input = net_neurons[layer-1][0].T

            # computing delta weights and updating weights for lth layer
            delta_weights = self.learning_rate * np.dot(mid_input, delta_hid)
            self.weights[layer] += delta_weights

            # updating biases for output layer
            delta_biases = self.learning_rate * delta_hid
            self.biases[layer] += delta_biases

    # train function gets optimal weights and biases matrix with train data
    def train(self, train_input, train_output):
        for iteration in range(self.epoch):
            for i in range(train_input.shape[0]):
                self.backpropagation(train_input[i], train_output[i])

    # Returns a predicted value for an input row
    def predict_classification(self, test_row):
        output = self.forward(test_row)[-1][0]
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
        # Adding accuracies to a list
        for i in range(k):
            index = [j * (i+1) for j in range(train_data.shape[0] // k)]
            train_data_clone = np.delete(train_data, index, 0)
            self.train(train_data_clone[:, :-1], train_data_clone[:, -1])
            accuracy = self.calculate_performance(validation_data[i][:, :-1], validation_data[i][:, -1])
            accuracy_list.append(accuracy)

        performance = sum(accuracy_list)*100/k
        return performance


def main():
    # Import train data
    input_path = path.join('..', 'Data', 'iris', 'iris_train.csv')
    data_in = np.loadtxt(input_path, delimiter=',')
    output_path = path.join('..', 'Data', 'iris', 'iris_train_label.csv')
    data_out = np.loadtxt(output_path, delimiter=',')
    data_train = np.column_stack((data_in, data_out))

    # Import test data
    test_input_path = path.join('..', 'Data', 'iris', 'iris_test.csv')
    data_test_in = np.loadtxt(test_input_path, delimiter=',')
    test_output_path = path.join('..', 'Data', 'iris', 'iris_test_label.csv')
    data_test_out = np.loadtxt(test_output_path, delimiter=',')

    # Training with data
    mlp = MLP()
    mlp.train(data_train[:, :-1], data_train[:, -1])

    # Make prediction and measure algorithm performance
    correct = 0
    for i in range(data_test_out.shape[0]):
        prediction = mlp.predict_classification(data_test_in[i])
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

    mlp = MLP()
    print('Performance with K fold validation:\n{:.2f}%'.format(mlp.K_fold_validation(data_train)))


if __name__ == '__main__':
    main()
