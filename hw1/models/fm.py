import collections
import re

import numpy as np
import torch
import torch.nn as nn

# полки-статусы
BAD_SHELF = re.compile(r"read|own|favou?rite|buy|wish|library|default|kindle|pdf|ebook|"
                       r"^poetry$|^poems?$|^shelf|^books?$|^my-|tbr|dnf|abandon|borrowed")


def item_tokens(books, split, n_shelves=10, min_books=30):
    # average_rating/ratings_count не берём
    b = books.set_index("book_id").loc[split.items]
    shelves = []
    for sh in b["popular_shelves"]:
        names = [x["name"] for x in sh if not BAD_SHELF.search(x["name"])]
        shelves.append(names[:n_shelves])
    cnt = collections.Counter(s for lst in shelves for s in lst)
    vocab = {s for s, c in cnt.items() if c >= min_books}

    toks = []
    for (bid, row), sh in zip(b.iterrows(), shelves):
        t = ["author=" + a["author_id"] for a in row["authors"][:3]]
        t += ["shelf=" + s for s in sh if s in vocab]
        y = int(row["publication_year"]) if row["publication_year"] else 0
        t.append("year=" + ("na" if y == 0 else "<1900" if y < 1900 else str(min(y, 2017) // 10 * 10)))
        p = int(row["num_pages"]) if row["num_pages"] else 0
        t.append("pages=" + ("na" if p == 0 else str(min(int(np.log2(p + 1)), 10))))
        lang = row["language_code"] or "na"
        t.append("lang=" + ("eng" if lang.startswith("en") else lang))
        t.append("format=" + (row["format"] or "na").lower())
        toks.append(t)
    return toks


def user_tokens(split):
    # Только по train
    tr = split.train
    n = tr.groupby("u").size().reindex(range(split.n_users)).values
    m = tr.groupby("u")["rating"].mean().reindex(range(split.n_users)).values
    return [["act=" + str(int(np.log2(a))), "mean=" + str(round(r * 2) / 2)] for a, r in zip(n, m)]


def pad(lists, vocab):
    L = max(len(x) for x in lists)
    out = np.zeros((len(lists), L), dtype=np.int64)  # 0 - паддинг
    for i, x in enumerate(lists):
        ids = [vocab[t] for t in x]
        out[i, :len(ids)] = ids
    return out


class FMNet(nn.Module):
    def __init__(self, n_feat, k):
        super().__init__()
        self.w = nn.Embedding(n_feat, 1, padding_idx=0)
        self.v = nn.Embedding(n_feat, k, padding_idx=0)
        nn.init.normal_(self.v.weight, 0, 0.01)
        nn.init.zeros_(self.w.weight)
        with torch.no_grad():
            self.v.weight[0] = 0

    def side(self, idx):
        # линейная часть, сумма векторов, взаимодействия внутри набора
        v = self.v(idx)
        s = v.sum(1)
        inner = 0.5 * ((s ** 2).sum(1) - (v ** 2).sum((1, 2)))
        return self.w(idx).sum((1, 2)), s, inner

    def forward(self, uidx, iidx):
        wu, su, iu = self.side(uidx)
        wi, si, ii = self.side(iidx)
        return wu + wi + iu + ii + (su * si).sum(1)


class FM:
    """FM, BPR"""

    def __init__(self, books, k=32, lr=0.01, reg=1e-5, epochs=10, batch=4096, use_meta=True, seed=42):
        self.books = books
        self.k, self.lr, self.reg, self.epochs, self.batch = k, lr, reg, epochs, batch
        self.use_meta = use_meta
        self.seed = seed

    def fit(self, split):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        ut = [["user=%d" % u] for u in range(split.n_users)]
        it = [["item=%d" % i] for i in range(split.n_items)]
        if self.use_meta:
            ut = [a + b for a, b in zip(ut, user_tokens(split))]
            it = [a + b for a, b in zip(it, item_tokens(self.books, split))]
        vocab = {}
        for t in [x for lst in ut + it for x in lst]:
            vocab.setdefault(t, len(vocab) + 1)
        self.n_feat = len(vocab)
        self.UF = torch.tensor(pad(ut, vocab))
        self.IF = torch.tensor(pad(it, vocab))

        self.net = FMNet(len(vocab) + 1, self.k)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        us = split.train["u"].values
        its = split.train["i"].values
        self.losses = []
        for ep in range(self.epochs):
            perm = rng.permutation(len(us))
            tot = 0
            for s in range(0, len(us), self.batch):
                b = perm[s:s + self.batch]
                u = torch.tensor(us[b])
                ip = torch.tensor(its[b])
                ineg = torch.tensor(rng.integers(0, split.n_items, len(b)))
                uf, pf, nf = self.UF[u], self.IF[ip], self.IF[ineg]
                diff = self.net(uf, pf) - self.net(uf, nf)
                reg = sum((self.net.v(x) ** 2).sum() for x in (uf, pf, nf))
                loss = -torch.nn.functional.logsigmoid(diff).sum() + self.reg * reg
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            self.losses.append(tot / len(us))

        # разложение для быстрого скоринга
        with torch.no_grad():
            wi, self.si, ii = self.net.side(self.IF)
            self.item_bias = (wi + ii).numpy()
            _, self.su, _ = self.net.side(self.UF)
            self.si, self.su = self.si.numpy(), self.su.numpy()
        return self

    def score(self, users):
        # члены только от пользователя на ранжирование не влияют
        return self.item_bias[None, :] + self.su[users] @ self.si.T
