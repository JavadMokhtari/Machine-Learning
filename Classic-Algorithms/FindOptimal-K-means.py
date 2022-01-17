import numpy as np
from matplotlib import pyplot as plt
from ClusteringValidityIndices import RMSSTD
from Kmeans import K_means


def find_optimal_K(input_dataset, max_K=10, iteration=100):
    K_values = list()
    for _iter in range(iteration):
        RMSSTD_values = list()
        for K in range(1, max_K):
            clustered = K_means(input_dataset, K)

            # Creating a list of clusters for CVI functions input
            clusters_list = list()
            for c in range(K):
                cluster = clustered[np.where(clustered[:, -1] == c)]
                clusters_list.append(cluster)

            # Getting RMSSTD index value
            cvi = RMSSTD(clusters_list)
            rmsstd = cvi.RMDDTD_index()
            RMSSTD_values.append(rmsstd)

        for i in range(1, len(RMSSTD_values)):
            if RMSSTD_values[i-1]/RMSSTD_values[i] < 1.15:
                optimal_K = i
                break
        K_values.append(optimal_K)

    # Setting figure of RMSSTD values by number of clusters
    # plt.plot([i for i in range(1, max_K)], RMSSTD_values)
    # plt.xlabel("Number of clusters")
    # plt.ylabel("RMSSTD")
    # plt.savefig('../Results/Elbow_method_figure.jpg')
    # plt.show()

    return max(set(K_values), key=K_values.count)


def main():
    # Load input data
    input_data = np.loadtxt('../Data/iris/iris_train.csv', delimiter=',')

    optimal_K = find_optimal_K(input_data)
    print("\nThe optimal value for K was found:\nK = {}".format(optimal_K))


if __name__ == "__main__":
    main()
