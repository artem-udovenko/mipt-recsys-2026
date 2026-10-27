import numpy as np


class ItemKNN:
    """Item-based kNN, косинус с shrinkage"""

    def __init__(self, k=100, shrink=10.0):
        self.k = k
        self.shrink = shrink

    def fit(self, split):
        X = split.X
        self.X = X
        G = (X.T @ X).toarray()
        norms = np.sqrt(np.diag(G))
        S = G / (np.outer(norms, norms) + self.shrink + 1e-8)
        np.fill_diagonal(S, 0)
        if self.k < S.shape[0]:
            low = np.argpartition(-S, self.k, axis=0)[self.k:]
            np.put_along_axis(S, low, 0, axis=0)
        self.S = S.astype(np.float32)
        return self

    def score(self, users):
        return self.X[users] @ self.S
