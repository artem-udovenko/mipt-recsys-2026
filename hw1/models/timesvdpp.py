import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class TimeSVDppNet(nn.Module):
    def __init__(self, n_users, n_items, n_bins, k):
        super().__init__()
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        self.bibin = nn.Embedding(n_items * n_bins, 1)  # b_{i,Bin(t)}
        self.au = nn.Embedding(n_users, 1)              # alpha_u
        self.p = nn.Embedding(n_users, k)
        self.ap = nn.Embedding(n_users, k)              # alpha_pu
        self.q = nn.Embedding(n_items, k)
        self.y = nn.Embedding(n_items, k)
        for e in [self.bu, self.bi, self.bibin, self.au, self.ap]:
            nn.init.zeros_(e.weight)
        for e in [self.p, self.q, self.y]:
            nn.init.normal_(e.weight, 0, 0.01)


class TimeSVDpp:
    """timeSVD++ без b_{u,t}.
    use_time=False - SVD++; neg_ratio - доля случайных пар с таргетом 0"""

    def __init__(self, k=32, lr=0.005, reg=0.02, epochs=15, n_bins=20, beta=0.4,
                 batch=16384, use_time=True, neg_ratio=0.0, seed=42):
        self.k, self.lr, self.reg, self.epochs = k, lr, reg, epochs
        self.n_bins, self.beta, self.batch = n_bins, beta, batch
        self.use_time = use_time
        self.neg_ratio = neg_ratio
        self.seed = seed

    def dev(self, u, day):
        d = day - self.tu[u]
        return np.sign(d) * np.abs(d) ** self.beta

    def get_bin(self, day):
        b = ((day - self.t0) / (self.t1 - self.t0 + 1e-9) * self.n_bins).astype(int)
        return np.clip(b, 0, self.n_bins - 1)

    def implicit(self, u):
        # |N(u)|^-1/2 sum y_j
        uniq, inv = np.unique(u, return_inverse=True)
        Nb = self.N[uniq].tocoo()
        Nt = torch.sparse_coo_tensor(np.vstack([Nb.row, Nb.col]), Nb.data, Nb.shape, check_invariants=False)
        return torch.sparse.mm(Nt, self.net.y.weight)[torch.tensor(inv)]

    def predict_parts(self, u, i, day):
        net = self.net
        dv = torch.tensor(self.dev(u, day) if self.use_time else np.zeros(len(u)), dtype=torch.float32)
        ut, it = torch.tensor(u), torch.tensor(i)
        pu = net.p(ut) + self.implicit(u)
        bias = self.mu + net.bu(ut)[:, 0] + net.bi(it)[:, 0]
        if self.use_time:
            bins = torch.tensor(self.get_bin(day) + i * self.n_bins)
            pu = pu + net.ap(ut) * dv[:, None]
            bias = bias + net.au(ut)[:, 0] * dv + net.bibin(bins)[:, 0]
        return bias + (net.q(it) * pu).sum(1)

    def fit(self, split):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        tr = split.train
        day = (tr["ts"].astype("int64") // 10 ** 9 / 86400).values
        u, i, r = tr["u"].values, tr["i"].values, tr["rating"].values.astype(np.float32)
        self.t0, self.t1 = day.min(), day.max()
        self.tu = tr.assign(day=day).groupby("u")["day"].mean().reindex(range(split.n_users)).values
        self.mu = float(r.mean()) / (1 + self.neg_ratio)

        # N(u) - оценённое в train
        X = split.X.tocsr()
        cnt = np.asarray(X.sum(1)).ravel()
        self.N = (sp.diags(1 / np.sqrt(np.maximum(cnt, 1))) @ X).astype(np.float32).tocsr()

        self.net = TimeSVDppNet(split.n_users, split.n_items, self.n_bins, self.k)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.losses = []
        for ep in range(self.epochs):
            perm = rng.permutation(len(r))
            tot = 0
            for s in range(0, len(r), self.batch):
                b = perm[s:s + self.batch]
                ub, ib, db, rb = u[b], i[b], day[b], r[b]
                if self.neg_ratio > 0:
                    nb = int(len(b) * self.neg_ratio)
                    j = rng.choice(b, nb)
                    ub = np.concatenate([ub, u[j]])
                    ib = np.concatenate([ib, rng.integers(0, split.n_items, nb)])
                    db = np.concatenate([db, day[j]])
                    rb = np.concatenate([rb, np.zeros(nb, dtype=np.float32)])
                pred = self.predict_parts(ub, ib, db)
                err = ((pred - torch.tensor(rb)) ** 2).sum()
                net = self.net
                ub, ib = torch.tensor(ub), torch.tensor(ib)
                reg = (net.p(ub) ** 2).sum() + (net.q(ib) ** 2).sum() + (net.bu(ub) ** 2).sum() \
                    + (net.bi(ib) ** 2).sum() + (net.ap(ub) ** 2).sum() + (net.au(ub) ** 2).sum() \
                    + (net.y.weight ** 2).sum() * len(ub) / len(r)
                loss = err + self.reg * reg
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += err.item()
            self.losses.append(np.sqrt(tot / (len(r) * (1 + self.neg_ratio))))
        return self

    def rmse(self, split, df):
        d = df[df["user_id"].isin(split.uid) & df["book_id"].isin(split.iid)]
        u = d["user_id"].map(split.uid).values
        i = d["book_id"].map(split.iid).values
        day = (d["ts"].astype("int64") // 10 ** 9 / 86400).values
        with torch.no_grad():
            p = self.predict_parts(u, i, day).clamp(1, 5).numpy()
        return float(np.sqrt(((p - d["rating"].values) ** 2).mean()))

    def score(self, users, day=None):
        # predict_parts на момент day для всех книг
        day = self.t1 if day is None else day
        users = np.asarray(users)
        net = self.net
        with torch.no_grad():
            ut = torch.tensor(users)
            pu = net.p(ut) + self.implicit(users)
            bi = net.bi.weight[:, 0].clone()
            bu = self.mu + net.bu(ut)[:, 0]
            if self.use_time:
                dv = torch.tensor(self.dev(users, day), dtype=torch.float32)
                pu = pu + net.ap(ut) * dv[:, None]
                bu = bu + net.au(ut)[:, 0] * dv
                b = int(self.get_bin(np.array([day]))[0])
                bi = bi + net.bibin.weight[b::self.n_bins, 0]
            return (bu[:, None] + bi[None, :] + pu @ net.q.weight.T).numpy()
