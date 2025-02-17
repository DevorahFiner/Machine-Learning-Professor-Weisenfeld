#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Your assignment is to complete the KMeans and DBSCAN classes below.

You are free to add any additional class methods or optional parameters,
but do not change the arguments that have been pre added.

Once done, copy and paste the classes in this cell into a text file and save it as: 
"usl_{college_id_num}.py" (replace {college_id_num} with your college id number) and submit.
For example, if your college id is 1234567, you should name your file usl_1234567.py
    - Do NOT inlcude the curly braces: {}
    - DO include the underscore: _
    - Do NOT capitalize usl
    - DO ensure that you are saving as a .py file
    - ONLY copy the text from this cell, DO NOT copy the text from other cells
    - DO test your code to ensure it works before you submit it.
"""

import numpy as np
class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_seed = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol # maximum change in centroid coordinates that are not counted as changed 
        self.centroids = None
        self.labels = None
        self.random_seed = random_seed

    def fit(self, X):
        # Randomly initialize the centroids by selecting n_clusters points from the dataset
        np.random.seed(self.random_seed)  # For reproducibility
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        # loop as many times as self.max_iter
            # Assign each point to the nearest centroid
            # Compute new centroids from the mean of the points in each cluster
            # Check for convergence (if centroids do not change more than the tolerance) and break if converged
        for _ in range(self.max_iter):
            # Assign each point to the nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)

            # Compute new centroids
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # Check for convergence
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break

            self.centroids = new_centroids

    def predict(self, X):
        # return array of cluster labels for each row of X (cluster labels should be integer from 0 to k-1)
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        # Initialize array of labels for each row of X (-1 means unclassified AKA noise)
        self.labels = np.full(X.shape[0], -1)  # Initialize all points as noise (-1)
        cluster_label = 0        
        #initialize first possible cluster label (0)
        # loop through each row of X
        for i in range(X.shape[0]):
            # if row is labeled -1, check to see if it is a core point, otherwise continue
                # if core point, label it with current cluster value and expand the cluster
                    # once the cluster is complete, increment the cluster label
            if self.labels[i] != -1:
                continue  # Skip already processed points

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                continue  # Label as noise if not a core point

            self.labels[i] = cluster_label  # Assign new cluster label to core point

            self._expand_cluster(X, neighbors, cluster_label)
            cluster_label += 1  # Move to the next cluster label
            
    def _get_neighbors(self, X, index):
        distances = np.sqrt(((X - X[index])**2).sum(axis=1))
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, neighbors, cluster_label):
        i = 0
        while i < len(neighbors):
            current_index = neighbors[i]
            
            if self.labels[current_index] == -1:
                self.labels[current_index] = cluster_label

            elif self.labels[current_index] == 0:
                self.labels[current_index] = cluster_label
                current_neighbors = self._get_neighbors(X, current_index)
                
                if len(current_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, current_neighbors))

            i += 1
                    
    def predict(self, X):
        # return array of cluster labels for each row of X (cluster labels should be integer from 0 to k-1)
        return self.labels

