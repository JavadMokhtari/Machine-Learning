import numpy as np
from matplotlib import pyplot as plt
from ClusteringValidityIndices import Dunn, Davies_Bouldin, RMSSTD
from os import path


# Is defined for calculating Euclidean distance between two rows of data
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = np.sqrt(distance)
    return distance


class Kmeans:
    def __init__(self):
        self.centers = None
        self.optimal_k = None
        self.clustered = None
        self.clusters_variance = None

    def K_means(self, dataset, K):
        dataset_size = dataset.shape[0]
        indx = np.random.choice(dataset_size, K, replace=False)
        self.centers = dataset[indx, :]
        new_centers = np.empty(self.centers.shape)
        self.clusters_variance = np.ones((self.centers.shape[0], 1))
        cluster_values = np.empty((dataset_size, 1))
        error_vector = np.empty(K)
        error = 1.000
        iteration = 0
        # print("\n\nFor K = {}:\n".format(K))

        while error > 0.005:
            for i in range(dataset_size):
                min_dist = float('inf')
                for j in range(K):
                    dist = euclidean_distance(dataset[i], self.centers[j, :])
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
                    new_center = np.array([np.sum(column_values[i]) / cluster.shape[0]
                                           for i in range(cluster.shape[1] - 1)])
                    distances = [euclidean_distance(cluster[i, :-1], new_center) for i in range(cluster.shape[0])]
                    self.clusters_variance[i][0] = sum(distances) / len(distances)
                new_centers[i, :] = new_center

            # Calculating distance between new centers and previous centers
            for i in range(K):
                error_vector[i] = euclidean_distance(new_centers[i, :], self.centers[i, :])
            error = np.sqrt(np.dot(error_vector, error_vector))

            iteration += 1
            # print("Relocation of centers in iteration {:2d}: {:.5f}".format(iteration, error))
            self.centers = np.copy(new_centers)

        # print('\n\n', 'The obtained centers are:', '\n\n', np.around(centers, decimals=2))
        self.clustered = clustered_dataset

    def find_optimal_K(self, input_dataset, max_K=10, iteration=30):
        K_values = list()
        for _iter in range(iteration):
            RMSSTD_values = list()
            for K in range(1, max_K):
                self.K_means(input_dataset, K)
                clustered = self.clustered

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
                if RMSSTD_values[i-1]/RMSSTD_values[i] <= 1.1:
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
    # import iris data and clustering
    input_path = path.join('..', 'Data', 'iris', 'iris_train.csv')
    input_data = np.loadtxt(input_path, delimiter=',')

    kmeans = Kmeans()
    optimal_K = kmeans.find_optimal_K(input_data)
    print("\nThe optimal K was found by RMSSTD index:\nK = {}".format(optimal_K))
    k = optimal_K
    kmeans.K_means(input_data, k)
    iris_clustered = kmeans.clustered

    # Creating a list of clusters for CVI functions input
    clusters_list = list()
    for c in range(k):
        cluster = iris_clustered[np.where(iris_clustered[:, -1] == c)]
        clusters_list.append(cluster)

    # Getting Dunn's index value
    cvi = Dunn(clusters_list)
    Dunn_value = cvi.Dun_index()
    print('\n\n', "Dunn's Index = {:.2f}".format(Dunn_value), sep='')

    # Getting Davies-Bouldin index value
    cvi = Davies_Bouldin(clusters_list)
    DB_value = cvi.Davies_Bouldin_index()
    print("Davies-Bouldin Index = {:.2f}".format(DB_value))

    # Visualization of clustering result
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x1 = iris_clustered[:, 0]
    x2 = iris_clustered[:, 1]
    x3 = iris_clustered[:, 2]
    x4 = iris_clustered[:, 3] * 20
    clusters_labels = iris_clustered[:, 4]

    img = ax.scatter(x1, x2, x3, s=x4, c=clusters_labels, cmap='turbo')
    fig.colorbar(img)
    # save_path = path.join('..', 'Results', 'K-means.scatter.jpg')
    # plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    main()
