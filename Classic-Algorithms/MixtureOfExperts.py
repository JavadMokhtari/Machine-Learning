import numpy as np
from math import sqrt


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


class MOE:
    def __init__(self, num_experts=5, lr_expert=0.01, lr_gate=0.01, epoch=5000):
        self.input_dimension = 4
        self.num_class = 3
        self.epoch = epoch
        self.lr_expert = lr_expert
        self.lr_gate = lr_gate
        self.num_experts = num_experts
        self.experts_weights = np.random.normal(0, 1, (self.num_experts, self.num_class, self.input_dimension))
        self.gate_weights = np.random.normal(0, 1, (self.num_experts, self.input_dimension))
        # self.experts_biases = np.random.normal(0, 1, self.num_experts)
        # self.gate_biases = np.random.normal(0, 1, self.num_experts)
        self.output = np.zeros((self.num_experts, self.num_class))
        self.gate_output = np.zeros(self.num_experts)
        self.gate = np.zeros(self.num_experts)

    def train(self, input_data, output_data):
        data_size = len(output_data)
        for i in range(data_size):
            for j in range(self.input_dimension):
                input_data[i, j] = input_data[i, j] / np.sum(input_data[:, j])
        # input_data /= 1000

        desired_output = np.empty((data_size, self.num_class))
        for i in range(data_size):
            output_row = np.zeros(self.num_class)
            output_row[int(output_data[i])] = 1
            desired_output[i] = output_row

        for epoch in range(self.epoch):
            for row in range(data_size):
                for i in range(self.num_experts):
                    for j in range(self.num_class):
                        self.output[i, j] = sigmoid(np.dot(input_data[row], self.experts_weights[i, j]))
                    self.gate_output[i] = np.dot(input_data[row], self.gate_weights[i])

                gate_summation = sum([np.exp(self.gate_output[j]) for j in range(self.num_experts)])

                for i in range(self.num_experts):
                    self.gate[i] = np.exp(self.gate_output[i]) / gate_summation

                error_summation = sum([self.gate[j] * np.exp(-0.5 * euclidean_distance(desired_output[row],
                                       self.output[j]) ** 2) for j in range(self.num_experts)])

                h = np.empty(self.gate.shape)
                delta_ew = np.empty(self.experts_weights.shape)
                delta_gw = np.empty(self.gate_weights.shape)

                for i in range(self.num_experts):
                    error = euclidean_distance(desired_output[row], self.output[i])
                    h[i] = self.gate[i] * (-0.5 * error ** 2) / error_summation

                    error = np.reshape(np.subtract(desired_output[row], self.output[i]), (3, 1))
                    input_row = np.reshape(input_data[row], (1, 4))
                    delta_ew = self.lr_expert * h[i] * np.dot(error, input_row)
                    delta_gw = self.lr_gate * (h[i] - self.gate[i]) * input_data[row]
                    # delta_eb = self.lr_expert * h[i] * (output_data[row] - self.output[i])
                    # delta_gb = self.lr_gate * (h[i] - self.gate[i])

                self.experts_weights += delta_ew
                self.gate_weights += delta_gw
                # self.experts_biases += delta_eb
                # self.experts_biases += delta_gb

            # print(self.experts_biases)
        # print(self.experts_weights)

    def predict_classification(self, input_vector):
        for i in range(self.num_experts):
            for j in range(self.num_class):
                self.output[i, j] = sigmoid(np.dot(input_vector, self.experts_weights[i, j]))
            self.gate_output[i] = np.dot(input_vector, self.gate_weights[i])

        gate_summation = sum([np.exp(self.gate_output[j]) for j in range(self.num_experts)])
        for i in range(self.num_experts):
            self.gate[i] = np.exp(self.gate_output[i]) / gate_summation

        total_output = np.dot(self.gate, self.output)
        return total_output


def main():
    # Import train data
    data_in = np.loadtxt('../Data/iris/iris_train.csv', delimiter=',')
    data_out = np.loadtxt('../Data/iris/iris_train_label.csv', delimiter=',')

    moe = MOE()
    moe.train(data_in, data_out)
    for i in range(data_in.shape[0]):
        print(np.exp(moe.predict_classification(data_in[i])))


if __name__ == "__main__":
    main()
