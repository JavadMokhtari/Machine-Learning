import numpy as np
from MultiLayerPerceptron import MLP
from KNearestNeighbor import KNN
from NaiveBayes import NaiveBayes
from ParzenWindow import ParzenWindow
from RadialBasisFunction import RBF
from os import path


class voting_combiner:
    def __init__(self, input_data, output_data):
        train_data = np.column_stack((input_data, output_data))
        self.num_experts = 5

        self.ex1 = MLP()
        self.ex1.train(input_data, output_data)

        self.ex2 = KNN(train_data, 10)
        self.ex3 = ParzenWindow(train_data, radius=0.7)

        self.ex4 = RBF(train_data)
        self.ex4.train()

        self.ex5 = NaiveBayes(train_data)

    def predict_classification(self, input_vector):
        experts_outputs = list()

        experts_outputs.append(self.ex1.predict_classification(input_vector))
        experts_outputs.append(self.ex2.predict_classification(input_vector))
        experts_outputs.append(self.ex3.predict_classification(input_vector))
        experts_outputs.append(self.ex4.predict_classification(input_vector))
        experts_outputs.append(self.ex5.predict_classification(input_vector))

        prediction = int(max(set(experts_outputs), key=experts_outputs.count))
        return prediction


def main():
    # Import train data
    input_path = path.join('..', 'Data', 'iris', 'iris_train.csv')
    input_data = np.loadtxt(input_path, delimiter=',')
    output_path = path.join('..', 'Data', 'iris', 'iris_train_label.csv')
    output_data = np.loadtxt(output_path, delimiter=',')

    # Import test data
    test_input_path = path.join('..', 'Data', 'iris', 'iris_test.csv')
    data_test_in = np.loadtxt(test_input_path, delimiter=',')
    test_output_path = path.join('..', 'Data', 'iris', 'iris_test_label.csv')
    data_test_out = np.loadtxt(test_output_path, delimiter=',')

    correct = 0
    static_combiner = voting_combiner(input_data, output_data)
    # With this loop, we classify all the test dataset row and measure performance
    for i in range(len(data_test_out)):
        prediction = static_combiner.predict_classification(data_test_in[i])
        real_value = data_test_out[i]
        print('For {} Expected {}, Got {}.'.format(data_test_in[i], real_value, prediction), end='  ')
        if real_value == prediction:
            correct += 1
            print('True')
        else:
            print('False')
    performance = correct / len(data_test_out)
    print("\nAccuracy on test dataset: {}%".format(performance * 100))


if __name__ == "__main__":
    main()
