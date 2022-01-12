# Self-Organizing Map (SOM) algorithm

import numpy as np
from math import sqrt


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    distance = sqrt(distance)
    return distance


