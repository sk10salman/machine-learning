import numpy as np

import pandas as pd
from matplotlib import pyplot as plt

from sklearn.datasets.samples_generator import make_blobs


def clusters(points, k):
    return points[np.random.randint(points.shape[0], size=k)]
    

def get_distances(centroid, points):
    return np.linalg.norm(points - centroid, axis=1)

X, Y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.show()

centroids = clusters(X, 4)

classes = np.zeros(X.shape[0], dtype=np.float64)
distances = np.zeros([X.shape[0], 4], dtype=np.float64)

for i in range(50):
    for i, c in enumerate(centroids):
        distances[:, i] = get_distances(c, X)
    classes = np.argmin(distances, axis=1)
    for c in range(4):
        centroids[c] = np.mean(X[classes == c], 0)
plt.scatter(X[:,0], X[:,1])
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red')
plt.show()