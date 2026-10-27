import numpy as np


def ndcg_at_k(rec, rel, k=10):
    dcg = sum(1 / np.log2(j + 2) for j, i in enumerate(rec[:k]) if i in rel)
    idcg = sum(1 / np.log2(j + 2) for j in range(min(len(rel), k)))
    return dcg / idcg


def recall_at_k(rec, rel, k=10):
    return len(set(rec[:k]) & rel) / min(len(rel), k)


def top_k(model, split, users, k=10, batch=2000):
    res = []
    for s in range(0, len(users), batch):
        u = users[s:s + batch]
        sc = np.asarray(model.score(u), dtype=np.float32).copy()
        seen = split.X[u]
        sc[seen.nonzero()] = -np.inf
        idx = np.argpartition(-sc, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sc, idx, axis=1), axis=1)
        res.append(np.take_along_axis(idx, order, axis=1))
    return np.vstack(res)


def evaluate(model, split, k=10):
    users = split.eval_users
    recs = top_k(model, split, users, k)
    nd = [ndcg_at_k(r, split.gt[u], k) for u, r in zip(users, recs)]
    rc = [recall_at_k(r, split.gt[u], k) for u, r in zip(users, recs)]
    cov = len(np.unique(recs)) / split.n_items
    return {"NDCG@10": np.mean(nd), "Recall@10": np.mean(rc), "Coverage@10": cov}


class Popular:
    def fit(self, split):
        self.pop = np.asarray(split.X.sum(0)).ravel()
        return self

    def score(self, users):
        return np.tile(self.pop, (len(users), 1))
