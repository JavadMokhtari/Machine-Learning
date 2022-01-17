import numpy as np
from math import log, pi


def get_mean(train_data, class_label):
    class_data = train_data[np.where(train_data[:, -1] == class_label)]
    column_values = [class_data[:, i] for i in range(class_data.shape[1] - 1)]
    mean_row = [np.sum(column_values[i]) / class_data.shape[0] for i in range(class_data.shape[1] - 1)]
    return mean_row


def dataset_cov(train_data, class_label):
    class_data = train_data[np.where(train_data[:, -1] == class_label)][:, :-1]
    d = class_data.shape[1]
    features = np.array([class_data[:, i] for i in range(d)])
    cov_matrix = np.cov(features)
    return cov_matrix


def multivariate_discrimination(train_data, class_label, test_row):
    prior = train_data[np.where(train_data[:, -1] == class_label)].shape[0]/train_data.shape[0]
    d = train_data.shape[1] - 1
    cov_matrix = dataset_cov(train_data, class_label)
    class_mean = get_mean(train_data, class_label)
    discrimination = log(prior) - 0.5 * d * log(2 * pi) - 0.5 * log(np.linalg.det(cov_matrix)) - 0.5 * \
        np.dot(np.dot(np.transpose(test_row - class_mean), np.linalg.inv(cov_matrix)), (test_row - class_mean))
    return discrimination


def predict_classification(train_data, test_row):
    train_outputs = set(train_data[:, -1])
    results = list()
    for class_label in train_outputs:
        discrimination = multivariate_discrimination(train_data, class_label, test_row)
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
