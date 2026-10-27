import gzip
import json
import os
import urllib.request

import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
FILES = {
    "goodreads_interactions_poetry.json.gz": BASE_URL + "byGenre/goodreads_interactions_poetry.json.gz",
    "goodreads_books_poetry.json.gz": BASE_URL + "byGenre/goodreads_books_poetry.json.gz",
    "goodreads_book_authors.json.gz": BASE_URL + "goodreads_book_authors.json.gz",
}


def download(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print("downloading", url)
            urllib.request.urlretrieve(url, path)


def read_json_gz(path, fields=None):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if fields:
                r = {k: r.get(k) for k in fields}
            rows.append(r)
    return pd.DataFrame(rows)


def load_interactions(data_dir="data"):
    df = read_json_gz(os.path.join(data_dir, "goodreads_interactions_poetry.json.gz"),
                      ["user_id", "book_id", "rating", "is_read", "date_added"])
    df["rating"] = df["rating"].astype(int)
    df["is_read"] = df["is_read"].astype(bool)
    df["ts"] = pd.to_datetime(df["date_added"], format="%a %b %d %H:%M:%S %z %Y",
                              errors="coerce", utc=True)
    return df.drop(columns="date_added")


def load_books(data_dir="data"):
    cols = ["book_id", "title", "authors", "popular_shelves", "publication_year",
            "num_pages", "language_code", "format", "is_ebook", "average_rating",
            "ratings_count", "description"]
    books = read_json_gz(os.path.join(data_dir, "goodreads_books_poetry.json.gz"), cols)
    authors = read_json_gz(os.path.join(data_dir, "goodreads_book_authors.json.gz"),
                           ["author_id", "name"])
    return books, authors


def k_core(df, min_user, min_item):
    while True:
        n = len(df)
        uc = df["user_id"].value_counts()
        df = df[df["user_id"].isin(uc[uc >= min_user].index)]
        ic = df["book_id"].value_counts()
        df = df[df["book_id"].isin(ic[ic >= min_item].index)]
        if len(df) == n:
            return df


def time_split(df, val_q=0.8, test_q=0.9):
    t_val = df["ts"].quantile(val_q)
    t_test = df["ts"].quantile(test_q)
    train = df[df["ts"] < t_val]
    val = df[(df["ts"] >= t_val) & (df["ts"] < t_test)]
    test = df[df["ts"] >= t_test]
    return train, val, test, t_val, t_test


class Split:
    # train + окно оценки, индексы по train

    def __init__(self, train, target, rel_thr=4):
        self.users = np.sort(train["user_id"].unique())
        self.items = np.sort(train["book_id"].unique())
        self.uid = {u: i for i, u in enumerate(self.users)}
        self.iid = {b: i for i, b in enumerate(self.items)}
        self.n_users, self.n_items = len(self.users), len(self.items)

        self.train = train.copy()
        self.train["u"] = self.train["user_id"].map(self.uid)
        self.train["i"] = self.train["book_id"].map(self.iid)

        self.R = sp.csr_matrix((self.train["rating"].astype(np.float32),
                                (self.train["u"], self.train["i"])),
                               shape=(self.n_users, self.n_items))
        self.X = (self.R > 0).astype(np.float32)

        # ground truth: тёплые пары, rating >= rel_thr, не из train
        t = target[target["user_id"].isin(self.uid) & target["book_id"].isin(self.iid)]
        t = t[t["rating"] >= rel_thr]
        t = t.assign(u=t["user_id"].map(self.uid), i=t["book_id"].map(self.iid))
        seen = set(zip(self.train["u"], self.train["i"]))
        t = t[[(a, b) not in seen for a, b in zip(t["u"], t["i"])]]
        self.gt = t.groupby("u")["i"].apply(set).to_dict()
        self.eval_users = np.array(sorted(self.gt))

        self.stats = {
            "target_rows": len(target),
            "cold_user_rows": int((~target["user_id"].isin(self.uid)).sum()),
            "cold_item_rows": int((target["user_id"].isin(self.uid) & ~target["book_id"].isin(self.iid)).sum()),
            "eval_users": len(self.eval_users),
            "relevant_pairs": int(sum(len(s) for s in self.gt.values())),
        }
