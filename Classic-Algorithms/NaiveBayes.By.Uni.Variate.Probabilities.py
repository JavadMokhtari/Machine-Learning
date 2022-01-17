import numpy as np
from math import log, pi


def dataset_mean(trian_data, class_label):
    class_data = trian_data[np.where(trian_data[:, -1] == class_label)]
    column_values = [class_data[:, i] for i in range(class_data.shape[1] - 1)]
    mean_row = [np.sum(column_values[i]) / class_data.shape[0] for i in range(class_data.shape[1] - 1)]
    return mean_row


def univariate_discrimination(train_data, class_label, test_row):
    class_data = train_data[np.where(train_data[:, -1] == class_label)]
    prior = train_data[np.where(train_data[:, -1] == class_label)].shape[0]/train_data.shape[0]
    d = class_data.shape[1] - 1
    class_mean = np.array(dataset_mean(train_data, class_label))
    discrimination = 1
    for i in range(d):
        dim_var = np.var(class_data[:, i])
        dim_std = np.std(class_data[:, i])
        discrimination = - 0.5 * log(2*pi) - 0.5 * log(dim_std) - np.subtract(test_row, class_mean)[i] ** 2 / (2 * dim_var)
    discrimination += log(prior)
    return discrimination


def predict_classification(train_data, test_row):
    train_outputs = set(train_data[:, -1])
    results = list()
    for class_label in train_outputs:
        discrimination = univariate_discrimination(train_data, class_label, test_row)
        results.append((class_label, discrimination))
    results.sort(key=lambda tup: tup[1])
    predict = results[-1][0]
    return predict


def main():
    # Import train data
    data_in = np.loadtxt('../Data/iris/iris_train.csv', delimiter=',')
    data_out = np.loadtxt('../Data/iris/iris_train_label.csv', delimiter=',')
    train_data = np.column_stack((data_in, data_out))
    # Import test data
    data_test_in = np.loadtxt('../Data/iris/iris_test.csv', delimiter=',')
    data_test_out = np.loadtxt('../Data/iris/iris_test_label.csv', delimiter=',')

    correct = 0
    # With this loop, we classify all the test dataset row and measure performance
    for i in range(len(data_test_out)):
        prediction = predict_classification(train_data, data_test_in[i])
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
