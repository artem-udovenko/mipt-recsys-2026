import numpy as np


class IALS:
    """iALS, c_ui = 1 + alpha * r_ui."""

    def __init__(self, factors=64, lam=10.0, alpha=1.0, iters=10, seed=42):
        self.factors = factors
        self.lam = lam
        self.alpha = alpha
        self.iters = iters
        self.seed = seed

    def _step(self, C, Y):
        f = Y.shape[1]
        YtY = Y.T @ Y
        out = np.zeros((C.shape[0], f), dtype=np.float64)
        for u in range(C.shape[0]):
            idx = C.indices[C.indptr[u]:C.indptr[u + 1]]
            c = C.data[C.indptr[u]:C.indptr[u + 1]]
            if len(idx) == 0:
                continue
            Yu = Y[idx]
            A = YtY + (Yu.T * c) @ Yu + self.lam * np.eye(f)
            b = Yu.T @ (1 + c)
            out[u] = np.linalg.solve(A, b)
        return out

    def fit(self, split, R=None):
        R = split.R if R is None else R
        C = (R * self.alpha).tocsr().astype(np.float64)
        Ct = C.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.U = rng.normal(0, 0.01, (C.shape[0], self.factors))
        self.V = rng.normal(0, 0.01, (C.shape[1], self.factors))
        for _ in range(self.iters):
            self.U = self._step(C, self.V)
            self.V = self._step(Ct, self.U)
        return self

    def loss(self, split):
        # только для маленьких данных
        P = (split.R > 0).toarray()
        Cf = 1 + self.alpha * split.R.toarray()
        E = P - self.U @ self.V.T
        return (Cf * E ** 2).sum() + self.lam * ((self.U ** 2).sum() + (self.V ** 2).sum())

    def score(self, users):
        return self.U[users] @ self.V.T
