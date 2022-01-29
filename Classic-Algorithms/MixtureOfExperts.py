import numpy as np
from os import path


# Activation function for experts
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MOE:
    def __init__(self, num_experts=3, lr_expert=0.01, lr_gate=0.01, epoch=500):
        # Initialization parameters
        self.input_dimension = 4
        self.input_norm = None
        self.num_class = 3
        self.epoch = epoch
        self.lr_expert = lr_expert
        self.lr_gate = lr_gate
        self.num_experts = num_experts
        # Setting weights for experts and gating network
        self.experts_weights = np.random.normal(0, 1, (self.num_experts, self.num_class, self.input_dimension))
        self.gate_weights = np.random.normal(0, 1, (self.num_experts, self.input_dimension))
        self.experts_biases = np.random.normal(0, 1, (self.num_experts, self.num_class))
        # Defining other matrix
        self.output = np.zeros((self.num_experts, self.num_class))
        self.error = np.empty((self.num_experts, self.num_class))
        self.gate_output = np.zeros(self.num_experts)
        self.gate = np.zeros(self.num_experts)

    def train(self, input_data, output_data):
        data_size = len(output_data)
        # Normalization of input data
        self.input_norm = np.linalg.norm(input_data)
        input_data /= self.input_norm

        # Creating output vector from output label
        desired_output = np.empty((data_size, self.num_class))
        for i in range(data_size):
            output_row = np.zeros(self.num_class)
            output_row[int(output_data[i])] = 1
            desired_output[i] = output_row

        for epoch in range(self.epoch):
            for row in range(data_size):
                # Getting experts outputs and gating network outputs
                for i in range(self.num_experts):
                    self.output[i] = sigmoid(np.dot(self.experts_weights[i], input_data[row]) + self.experts_biases[i])
                    self.gate_output[i] = np.dot(input_data[row], self.gate_weights[i])

                # Applying soft max function on gating network outputs
                gate_summation = np.sum(np.array([np.exp(self.gate_output[j]) for j in range(self.num_experts)]))
                for i in range(self.num_experts):
                    self.gate[i] = np.exp(self.gate_output[i]) / gate_summation

                    # Calculating error for i-th expert
                    self.error[i] = np.subtract(desired_output[row], self.output[i])

                h = np.empty(self.gate.shape)
                errors_summation = np.sum(np.array([self.gate[j] * np.exp(-0.5 * np.dot(self.error[j].T, self.error[j]))
                                                    for j in range(self.num_experts)]))

                # Updating experts weights and gating network weights
                for i in range(self.num_experts):
                    h[i] = self.gate[i] * (-0.5 * np.dot(self.error[i].T, self.error[i])) / errors_summation

                    # Getting delta weights and delta biases for experts
                    delta_ew = self.lr_expert * h[i] * np.dot(np.reshape(self.error[i], (3, 1)),
                                                              np.reshape(input_data[row], (1, 4)))
                    delta_eb = self.lr_expert * h[i] * self.error[i]

                    # The expert with lower error gets more weight in gating network
                    delta_gw = self.lr_gate * (h[i] - self.gate[i]) * input_data[row]

                    self.experts_weights[i] += delta_ew
                    self.gate_weights[i] += delta_gw
                    self.experts_biases[i] += delta_eb

    # Predict output value for input vector
    def predict_classification(self, input_vector):
        normalized_input_vector = input_vector / self.input_norm

        for i in range(self.num_experts):
            self.output[i] = sigmoid(np.dot(self.experts_weights[i], normalized_input_vector) + self.experts_biases[i])
            self.gate_output[i] = np.dot(normalized_input_vector, self.gate_weights[i])

        gate_summation = np.sum(np.array([np.exp(self.gate_output[j]) for j in range(self.num_experts)]))
        for i in range(self.num_experts):
            self.gate[i] = np.exp(self.gate_output[i]) / gate_summation

        total_output = list(map(round, np.dot(self.gate, self.output)))
        for output in total_output:
            if output == 1:
                return total_output.index(output)
        return 0


def main():
    # Import train data
    input_path = path.join('..', 'Data', 'iris', 'iris_train.csv')
    input_data = np.loadtxt(input_path, delimiter=',')
    output_path = path.join('..', 'Data', 'iris', 'iris_train_label.csv')
    output_data = np.loadtxt(output_path, delimiter=',')

    # Import test data
    test_input_path = path.join('..', 'Data', 'iris', 'iris_test.csv')
    input_test_data = np.loadtxt(test_input_path, delimiter=',')
    test_output_path = path.join('..', 'Data', 'iris', 'iris_test_label.csv')
    output_test_data = np.loadtxt(test_output_path, delimiter=',')

    # Training with data
    moe = MOE()
    moe.train(input_data, output_data)

    # Predicting output on test data and getting performance
    correct = 0
    for i in range(output_test_data.shape[0]):
        prediction = moe.predict_classification(input_test_data[i])
        real_value = output_test_data[i]
        print('For {} Expected {}, Got {}.'.format(input_test_data[i], real_value, prediction), end='  ')
        if real_value == prediction:
            correct += 1
            print('True')
        else:
            print('False')
    performance = correct / output_test_data.shape[0]
    print("\nAccuracy on test dataset: {:.2f}%".format(performance * 100))


if __name__ == "__main__":
    main()
