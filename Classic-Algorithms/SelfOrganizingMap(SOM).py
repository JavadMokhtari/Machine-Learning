# Self-Organizing Map (SOM) algorithm

import numpy as np
from math import sqrt


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = sqrt(distance)
    return distance


class SelfOrganizingMap:

    def __int__(self, num_input=4, output_layer=None, learning_rate=0.01, epoch=2000):

        # output_layer is a list that contains number of neurons in this layer
        if output_layer is None:
            output_layer = [5, 5]

        self.num_input = num_input
        self.num_output = output_layer
        self.learning_rate = learning_rate
        self.epoch = epoch

        self.weights = np.random.normal(0, 1, tuple(output_layer))
        print(self.weights)
        self.biases = np.random.normal(0, 1, )
        print(self.biases)


def main():
    som = SelfOrganizingMap()
    print()


if __name__ == "__main__":
    main()
