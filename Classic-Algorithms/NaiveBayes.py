import numpy as np
from os import path


class NaiveBayes:
    def __init__(self, train_data):
        self.train_data = train_data
        self.mean_row = None
        self.cov_matrix = None

    def get_mean(self, class_label):
        class_data = self.train_data[np.where(self.train_data[:, -1] == class_label)]
        column_values = [class_data[:, i] for i in range(class_data.shape[1] - 1)]
        self.mean_row = [np.sum(column_values[i]) / class_data.shape[0] for i in range(class_data.shape[1] - 1)]

    def dataset_cov(self, class_label):
        class_data = self.train_data[np.where(self.train_data[:, -1] == class_label)][:, :-1]
        d = class_data.shape[1]
        features = np.array([class_data[:, i] for i in range(d)])
        self.cov_matrix = np.cov(features)

    def multivariate_discrimination(self, class_label, test_row):
        prior = self.train_data[np.where(self.train_data[:, -1] == class_label)].shape[0] / self.train_data.shape[0]
        d = self.train_data.shape[1] - 1
        self.dataset_cov(class_label)
        self.get_mean(class_label)
        discrimination = np.log(prior) - 0.5 * d * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(self.cov_matrix)) - \
            0.5 * np.dot(np.dot(np.transpose(test_row - self.mean_row), np.linalg.inv(self.cov_matrix)),
                         (test_row - self.mean_row))
        return discrimination

    def predict_classification(self, test_row):
        train_outputs = set(self.train_data[:, -1])
        results = list()
        for class_label in train_outputs:
            discrimination = self.multivariate_discrimination(class_label, test_row)
            results.append((class_label, discrimination))
        results.sort(key=lambda tup: tup[1])
        predict = results[-1][0]
        return predict


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

    naivebayes = NaiveBayes(train_data)
    correct = 0
    # With this loop, we classify all the test dataset row and measure performance
    for i in range(len(data_test_out)):
        prediction = naivebayes.predict_classification(data_test_in[i])
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
