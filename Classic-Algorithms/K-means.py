import numpy as np
from math import sqrt
from matplotlib import pyplot as plt


K = 3


# Is defined for calculating Euclidean distance between two row of data
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = sqrt(distance)
    return distance


# Returns mean of data as center
def set_centers(cluster_data):
    column_values = [cluster_data[:, i] for i in range(cluster_data.shape[1] - 1)]
    center = np.array([np.sum(column_values[i]) / cluster_data.shape[0] for i in range(cluster_data.shape[1] - 1)])
    return center


def K_means(dataset):
    dataset_size = dataset.shape[0]
    indx = np.random.choice(dataset_size, K, replace=False)
    centers = dataset[indx, :]
    new_centers = np.empty(centers.shape)
    cluster_values = np.empty((dataset_size, 1))
    error_vector = np.empty(K)
    error = 1
    epoch = 0

    while error > 0.005:
        for i in range(dataset_size):
            min_dist = 10 ** 5
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
            new_centers[i, :] = set_centers(cluster)

        # Calculating distance between new centers and previous centers
        for i in range(K):
            error_vector[i] = euclidean_distance(new_centers[i, :], centers[i, :])
        error = sqrt(np.dot(error_vector, error_vector))

        epoch += 1
        print("The amount of centers changing in round {:2d}: {:.5f}".format(epoch, error))
        centers = np.copy(new_centers)

    return clustered_dataset


def main():
    # import iris data and clustering
    data = np.loadtxt('../Data/iris/iris_train.csv', delimiter=',')
    iris_clustered = K_means(data)

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
    plt.savefig('../Results/K-means.scatter.jpg')
    plt.show()


if __name__ == '__main__':
    main()
