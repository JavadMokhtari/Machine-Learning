import numpy as np


# Is defined for calculating Euclidean distance between two rows of data
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = np.sqrt(distance)
    return distance


# Returns mean row of cluster as center
def get_center(cluster):
    column_values = [cluster[:, i] for i in range(cluster.shape[1])]
    center = [np.sum(column_values[i]) / cluster.shape[0] for i in range(cluster.shape[1])]
    return center


##############################################
# Root Means Square Standard Deviation Index #
##############################################
class RMSSTD:
    def __init__(self, clusters_list):
        self.clusters = clusters_list

    # Suppose set D contains n nodes with p dimensions. Sum of square parameter obtain as following function
    @staticmethod
    def sum_of_square(nodes_group):
        center = get_center(nodes_group)
        SS = 0
        for node in nodes_group:
            dist_square = euclidean_distance(node, center) ** 2
            SS += dist_square
        return SS

    def RMDDTD_index(self):
        K = len(self.clusters)
        SS_values = list()
        n = 0
        p = self.clusters[0].shape[1]
        for i in range(K):
            SSi = self.sum_of_square(self.clusters[i])
            SS_values.append(SSi)
            n += self.clusters[i].shape[0]

        RMSSTD_value = np.sqrt(sum(SS_values) / (p * (n - K)))
        return RMSSTD_value


################
# Dunn's Index #
################
class Dunn:
    # Returns Dunn's index value with getting a list from clusters in input
    def __init__(self, clusters_list):
        # Initializing parameters
        self.clusters = clusters_list
        self.separability = 0
        self.compactness = float('inf')

    # The first parameter of Dunn's index
    def separability_calculation(self, cluster1, cluster2):
        dist_list = list()
        for row1 in cluster1:
            for row2 in cluster2:
                distance = euclidean_distance(row1, row2)
                dist_list.append(distance)
        if dist_list:
            self.separability = min(dist_list)
        else:
            self.separability = 0

    # The second parameter of Dunn's index
    def compactness_calculation(self, cluster):
        dist_list = list()
        for i in range(cluster.shape[0]):
            for j in range(i + 1, cluster.shape[0]):
                distance = euclidean_distance(cluster[i], cluster[j])
                dist_list.append(distance)
        if dist_list:
            self.compactness = max(dist_list)
        else:
            self.compactness = float('inf')

    # Calculating Dunn's index value
    def Dun_index(self):
        separability_list = list()
        K = len(self.clusters)
        for i in range(K):
            for j in range(i+1, K):
                self.separability_calculation(self.clusters[i], self.clusters[j])
                separability_list.append(self.separability)

        compactness_list = list()
        for i in range(K):
            self.compactness_calculation(self.clusters[i])
            compactness_list.append(self.compactness)

        Dunn_value = min(separability_list) / max(compactness_list)
        return Dunn_value


########################
# Davies-Bouldin Index #
########################
class Davies_Bouldin:
    # Returns Davies-Bouldin index value with getting a list from clusters in input
    def __init__(self, clusters_list):
        # Initializing parameters
        self.clusters = clusters_list
        self.dispersion = float('inf')
        self.dissimilarity = 0

    # The first parameter of Davies-Bouldin index
    def dispersion_calculation(self, cluster):
        dist_sum = 0
        center = get_center(cluster)
        for row in cluster:
            dist = euclidean_distance(row, center)
            dist_sum += dist
        self.dispersion = dist_sum / cluster.shape[0]

    # The second parameter of Davies-Bouldin index
    def dissimilarity_calculation(self, cluster1, cluster2):
        center1 = get_center(cluster1)
        center2 = get_center(cluster2)
        self.dissimilarity = euclidean_distance(center1, center2)

    # Calculating Davies-Bouldin index value
    def Davies_Bouldin_index(self):
        Rij_values = list()
        Sn_values = list()
        K = len(self.clusters)
        for i in range(K):
            self.dispersion_calculation(self.clusters[i])
            Si = self.dispersion
            Sn_values.append(Si)

        for i in range(K):
            Ri_values = list()
            for j in range(K):
                if i == j:
                    continue
                self.dissimilarity_calculation(self.clusters[i], self.clusters[j])
                Dij = self.dissimilarity
                Rij = (Sn_values[i] + Sn_values[j]) / Dij
                Rij_values.append(Rij)
            if Rij_values:
                Ri = max(Rij_values)
                Ri_values.append(Ri)
        if Ri_values:
            DB_value = sum(Ri_values) / K
            return DB_value
        else:
            return float('inf')
