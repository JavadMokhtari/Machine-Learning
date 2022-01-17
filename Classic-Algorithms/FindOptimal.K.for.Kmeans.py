import numpy as np
from matplotlib import pyplot as plt
from ClusteringValidityIndices import RMSSTD
from Kmeans import K_means



def main():
    # Load input data
    input_data = np.loadtxt('../Data/iris/iris_train.csv', delimiter=',')

    optimal_K = find_optimal_K(input_data)
    print("\nThe optimal value for K was found:\nK = {}".format(optimal_K))


if __name__ == "__main__":
    main()
