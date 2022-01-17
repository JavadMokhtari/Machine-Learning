import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from ClusteringValidityIndices import Dunn, Davies_Bouldin, RMSSTD


# Is defined for calculating Euclidean distance between two rows of data
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = sqrt(distance)
    return distance


def K_means(dataset, K):
    dataset_size = dataset.shape[0]
    indx = np.random.choice(dataset_size, K, replace=False)
    centers = dataset[indx, :]
    new_centers = np.empty(centers.shape)
    cluster_values = np.empty((dataset_size, 1))
    error_vector = np.empty(K)
    error = 1.000
    iteration = 0
    # print("\n\nFor K = {}:\n".format(K))

    while error > 0.005:
        for i in range(dataset_size):
            min_dist = float('inf')
            for j in range(K):
                dist = euclidean_distance(dataset[i], centers[j, :])
                if dist <= min_dist:
                    min_dist = dist
                    cluster_values[i, :] = j
        # Set a cluster to each dataset row
        clustered_dataset = np.column_stack((dataset, cluster_values))

        # Updating centers
        for i in range(K):
            cluster = clustered_dataset[np.where(clustered_dataset[:, -1] == i)]

            if cluster.shape[0] == 0:
                indx = np.random.randint(dataset_size)
                new_center = dataset[indx, :]
            else:
                column_values = [cluster[:, i] for i in range(cluster.shape[1] - 1)]
                new_center = np.array(
                    [np.sum(column_values[i]) / cluster.shape[0] for i in range(cluster.shape[1] - 1)])

            new_centers[i, :] = new_center

        # Calculating distance between new centers and previous centers
        for i in range(K):
            error_vector[i] = euclidean_distance(new_centers[i, :], centers[i, :])
        error = sqrt(np.dot(error_vector, error_vector))

        iteration += 1
        # print("Relocation of centers in iteration {:2d}: {:.5f}".format(iteration, error))
        centers = np.copy(new_centers)

    # print('\n\n', 'The obtained centers are:', '\n\n', np.around(centers, decimals=2))
    return clustered_dataset


def main():
    # import iris data and clustering
    input_data = np.loadtxt('../Data/iris/iris_train.csv', delimiter=',')

    k = 3
    iris_clustered = K_means(input_data, k)

    # Creating a list of clusters for CVI functions input
    clusters_list = list()
    for c in range(k):
        cluster = iris_clustered[np.where(iris_clustered[:, -1] == c)]
        clusters_list.append(cluster)

    # Getting Dunn's index value
    cvi = Dunn(clusters_list)
    Dunn_value = cvi.Dun_index()

    # Getting Davies-Bouldin index value
    cvi = Davies_Bouldin(clusters_list)
    DB_value = cvi.Davies_Bouldin_index()

    # I consider an index that is obtained by dividing the Dunn's index by Davies-Bouldin index
    my_index = Dunn_value / DB_value
    print('\n\n', "Clustering Validity Index Value is: {:.2f}\n".format(my_index), sep='')
    print("This index is obtained by dividing the Dunn's index by Davies-Bouldin index")

    # visualization of clustering result
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x1 = iris_clustered[:, 0]
    x2 = iris_clustered[:, 1]
    x3 = iris_clustered[:, 2]
    x4 = iris_clustered[:, 3] * 20
    clusters_labels = iris_clustered[:, 4]

    img = ax.scatter(x1, x2, x3, s=x4, c=clusters_labels, cmap='turbo')
    fig.colorbar(img)
    # plt.savefig('../Results/K-means.scatter.jpg')
    plt.show()


if __name__ == '__main__':
    main()
