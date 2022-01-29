import numpy as np
from MultiLayerPerceptron import MLP
from KNearestNeighbor import KNN
from os import path


class Stack_Generalization:
    def __init__(self):
        self.num_experts = 4
        self.experts = None
        self.out_expert = None

    def train(self, input_dataset, output_dataset):
        num_data = len(output_dataset)
        train = np.column_stack((input_dataset, output_dataset))

        ex1 = MLP(hidden_layers=[5])
        ex2 = MLP(hidden_layers=[6], learning_rate=0.02)
        ex3 = MLP(hidden_layers=[4], learning_rate=0.006, epoch=2000)
        ex4 = KNN(train, 10)

        self.experts = [ex1, ex2, ex3, ex4]
        experts_output = np.zeros(self.num_experts)
        experts_outputs = np.zeros((num_data, self.num_experts))

        for i in range(self.num_experts-1):
            self.experts[i].train(input_dataset, output_dataset)

        for i in range(num_data):
            for j in range(self.num_experts):
                experts_output[j] = self.experts[j].predict_classification(input_dataset[i, :])
            experts_outputs[i, :] = experts_output

        self.out_expert = MLP()
        self.out_expert.train(experts_outputs, output_dataset)

    def predict(self, input_vector):
        experts_output = np.zeros(self.num_experts)
        for i in range(self.num_experts):
            experts_output[i] = self.experts[i].predict_classification(input_vector)
        predict = self.out_expert.predict_classification(experts_output)
        return predict


def main():
    # Import train data
    input_path = path.join('..', 'Data', 'iris', 'iris_train.csv')
    data_in = np.loadtxt(input_path, delimiter=',')
    output_path = path.join('..', 'Data', 'iris', 'iris_train_label.csv')
    data_out = np.loadtxt(output_path, delimiter=',')

    stack_generalization = Stack_Generalization()
    stack_generalization.train(data_in, data_out)

    # Import test data
    test_input_path = path.join('..', 'Data', 'iris', 'iris_test.csv')
    data_test_in = np.loadtxt(test_input_path, delimiter=',')
    test_output_path = path.join('..', 'Data', 'iris', 'iris_test_label.csv')
    data_test_out = np.loadtxt(test_output_path, delimiter=',')

    # Make prediction and measure algorithm performance
    correct = 0
    for i in range(data_test_out.shape[0]):
        prediction = stack_generalization.predict(data_test_in[i])
        real_value = data_test_out[i]
        print('For {} Expected {}, Got {}.'.format(data_test_in[i], real_value, prediction), end='  ')
        if real_value == prediction:
            correct += 1
            print('True')
        else:
            print('False')
    performance = correct / data_test_out.shape[0]
    print("\nAccuracy on test dataset: {:.2f}%".format(performance * 100))


if __name__ == "__main__":
    main()
