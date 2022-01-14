import numpy as np
from math import sqrt


# Is defined for calculating Euclidean distance between two rows of data
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = sqrt(distance)
    return distance


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
        self.separability = min(dist_list)

    # The second parameter of Dunn's index
    def compactness_calculation(self, cluster):
        dist_list = list()
        for i in range(cluster.shape[0]):
            for j in range(i + 1, cluster.shape[0]):
                distance = euclidean_distance(cluster[i], cluster[j])
                dist_list.append(distance)
        self.compactness = max(dist_list)

    # Calculating Dunn's index value
    def Dun_index(self):
        separability_list = list()
        for i in range(len(self.clusters)):
            for j in range(i+1, len(self.clusters)):
                self.separability_calculation(self.clusters[i], self.clusters[j])
                separability_list.append(self.separability)

        compactness_list = list()
        for i in range(len(self.clusters)):
            self.compactness_calculation(self.clusters[i])
            compactness_list.append(self.compactness)

        Dunn_value = min(separability_list) / max(compactness_list)
        return Dunn_value
