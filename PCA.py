import numpy as np


class PCA:
    def __init__(self, final_dim) -> None:
        self.final_dim = final_dim
        self.eigen_values = None
        self.eigen_vectors = None


    def fit(self, X):
        cov_matrix = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        projected_data = X.dot(eigenvectors[:, :self.final_dim])
        self.eigen_values = eigenvalues
        self.eigen_vectors = eigenvectors

        return projected_data


    def transform(self, X):
        return X.dot(self.eigen_vectors[:, :self.final_dim])
