import numpy as np
from matplotlib import pyplot as plt
from os import path


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = np.sqrt(distance)
    return distance


# Returns degree of neighborhood with respect to winner
def neighborhood_coefficient(node1, node2):
    return np.exp(-euclidean_distance(node1, node2)**2/2)


class SelfOrganizingMap:

    def __init__(self, num_input=4, num_output=None, epoch=1000):

        # output_layer is a list that contains number of neurons in this layer
        if num_output is None:
            self.num_output = (1, 3)

        self.num_input = num_input
        self.learning_rate = 0.8/epoch
        self.epoch = epoch

        # Initialization of weight matrix
        np.random.seed(1)
        self.weights = np.empty(self.num_output + (self.num_input,))
        for i in range(self.num_output[0]):
            for j in range(self.num_output[1]):
                self.weights[i][j] = np.random.normal(0, 1, (1, self.num_input))

    # Weights learn the input values
    def train(self, input_dataset):
        for epoch in range(self.epoch, 0, -1):
            learning_rate = self.learning_rate * epoch
            for input_row in input_dataset:
                distance_list = list()
                for i in range(self.num_output[0]):
                    for j in range(self.num_output[1]):
                        distance = euclidean_distance(input_row, self.weights[i][j])
                        distance_list.append((i, j, distance))
                distance_list.sort(key=lambda tup: tup[-1])
                BMU = distance_list[0]

                # Calculating delta_weights and updating weights
                delta_weights = np.empty(self.weights.shape)
                for i in range(self.num_output[0]):
                    for j in range(self.num_output[1]):
                        delta_weights[i][j] = learning_rate * neighborhood_coefficient((i, j), BMU[:2]) * \
                                              np.subtract(input_row, self.weights[i][j])
                self.weights += delta_weights

    # We set a label for each row of data in dataset
    def set_cluster(self, input_dataset):
        cluster_values = np.empty((input_dataset.shape[0], 1))
        for k in range(input_dataset.shape[0]):
            min_dist = 10 ** 5
            for i in range(self.num_output[0]):
                for j in range(self.num_output[1]):
                    dist = euclidean_distance(input_dataset[k], self.weights[i][j])
                    if dist <= min_dist:
                        min_dist = dist
                        cluster_values[k][0] = j

        clustered_dataset = np.column_stack((input_dataset, cluster_values))
        return clustered_dataset


def main():
    # Import data
    input_path = path.join('..', 'Data', 'iris', 'iris_train.csv')
    input_data = np.loadtxt(input_path, delimiter=',')

    som = SelfOrganizingMap()
    som.train(input_data)
    iris_clustered = som.set_cluster(input_data)

    # visualization of clustering result
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    x1 = iris_clustered[:, 0]
    x2 = iris_clustered[:, 1]
    x3 = iris_clustered[:, 2]
    x4 = iris_clustered[:, 3] * 50
    clusters = iris_clustered[:, 4]

    img = ax.scatter(x1, x2, x3, s=x4, c=clusters, cmap='turbo')
    fig.colorbar(img)
    plt.show()
    save_path = path.join('..', 'Results', 'SOM.scatter.jpg')
    plt.savefig(save_path)


if __name__ == "__main__":
    main()
