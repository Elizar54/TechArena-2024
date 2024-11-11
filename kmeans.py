import numpy as np


class My_KMeans:
    '''
    A custom implementation of the KMeans clustering algorithm.

    This class allows users to fit a KMeans model to a dataset and predict cluster labels for new data points.
    It includes methods for fitting the model to the data, predicting cluster labels, and computing distances 
    between data points and centroids.

    Parameters:
    ----------
    n_clusters : int, optional
        The number of clusters to form. Default is 3.
        
    max_iters : int, optional
        The maximum number of iterations for the algorithm to run. Default is 100.
        
    random_state : int, optional
        Controls the randomness of the initial centroid selection. Default is None.

    Attributes:
    ----------
    centroids : ndarray, shape (n_clusters, n_features)
        The final positions of the centroids after fitting the model.

    Methods:
    -------
    fit(X)
        Computes the KMeans clustering on the provided dataset X.
        
    predict(X)
        Predicts the closest cluster for each sample in X.
        
    _compute_distances(X)
        Computes the Euclidean distances between each point in X and the centroids.
    '''
    
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None


    def fit(self, X):
        np.random.seed(self.random_state)
            
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        return labels


    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)


    def _compute_distances(self, X):   
        return np.sqrt(((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))