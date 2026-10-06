"""SearchForge · 稠密检索（FAISS + LSI 嵌入；缺失则 numpy 精确余弦兜底）。

星 (晨星) · 2026-10-07

稠密表示 = TruncatedSVD(lsi_dim) 作用于 TF-IDF（即经典 LSI / 潜在语义索引）。
FAISS(IndexFlatIP) 在归一化向量上做内积 = 余弦相似度，精确且确定性。
离线兜底：同样 LSI 嵌入，但用 numpy argpartition 取 top-k（结果一致）。
"""

from __future__ import annotations

import numpy as np

from ..core.types import Corpus, Query
from ._tfidf import TfidfIndex


def _fit_lsi(doc_vecs: np.ndarray, lsi_dim: int, random_state: int):
    """拟合并返回 (embeddings, 归一化; svd 对象)。"""
    from sklearn.decomposition import TruncatedSVD

    n_samples, n_features = doc_vecs.shape
    dim = min(lsi_dim, n_features - 1, n_samples - 1)
    dim = max(2, dim)
    svd = TruncatedSVD(n_components=dim, random_state=random_state)
    emb = svd.fit_transform(doc_vecs)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    return emb, svd, dim


def _vocab_size(corpus: Corpus) -> int:
    return int(max((max(d.terms) for d in corpus.docs if d.terms), default=0)) + 1


class DenseRetriever:
    """FAISS 稠密检索器（LSI 嵌入 + 内积近邻）。"""

    name = "dense_faiss"

    def __init__(self, lsi_dim: int = 100, random_state: int = 42) -> None:
        self.lsi_dim = lsi_dim
        self.random_state = random_state
        self._tfidf = None
        self._emb = None
        self._svd = None
        self._index = None

    @staticmethod
    def available() -> bool:
        try:
            import faiss  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, corpus: Corpus) -> "DenseRetriever":
        vocab_size = _vocab_size(corpus)
        self._tfidf = TfidfIndex(vocab_size)
        doc_vecs = self._tfidf.fit(corpus).transform_docs(corpus)
        self._emb, self._svd, self.lsi_dim = _fit_lsi(doc_vecs, self.lsi_dim, self.random_state)
        if self.available():
            import faiss

            self._index = faiss.IndexFlatIP(self.lsi_dim)
            self._index.add(self._emb.astype(np.float32))
        return self

    def _query_emb(self, query: Query) -> np.ndarray:
        qv = self._tfidf.transform_queries([query])  # [1, V]
        emb = self._svd.transform(qv)  # [1, dim]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32).reshape(1, -1)

    def scores(self, query: Query) -> np.ndarray:
        qe = self._query_emb(query)
        return self._emb @ qe.reshape(-1)  # [n_docs]

    def search(self, query: Query, top_k: int) -> list[int]:
        qe = self._query_emb(query)
        if self._index is not None:
            k = min(top_k, self._emb.shape[0])
            _, idx = self._index.search(qe, k)
            return [int(i) for i in idx[0] if i >= 0]
        s = self.scores(query)
        k = min(top_k, len(s))
        idx = np.argpartition(-s, range(k))[:k]
        idx = idx[np.argsort(-s[idx])]
        return [int(i) for i in idx]


class NumpyCosineRetriever:
    """离线兜底（无 FAISS）：同样 LSI 嵌入 + numpy 精确 top-k。"""

    name = "dense_numpy"

    def __init__(self, lsi_dim: int = 100, random_state: int = 42) -> None:
        self.lsi_dim = lsi_dim
        self.random_state = random_state
        self._tfidf = None
        self._emb = None
        self._svd = None

    @staticmethod
    def available() -> bool:
        return True

    def fit(self, corpus: Corpus) -> "NumpyCosineRetriever":
        vocab_size = _vocab_size(corpus)
        self._tfidf = TfidfIndex(vocab_size)
        doc_vecs = self._tfidf.fit(corpus).transform_docs(corpus)
        self._emb, self._svd, self.lsi_dim = _fit_lsi(doc_vecs, self.lsi_dim, self.random_state)
        return self

    def _query_emb(self, query: Query) -> np.ndarray:
        qv = self._tfidf.transform_queries([query])
        emb = self._svd.transform(qv)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32).reshape(1, -1)

    def scores(self, query: Query) -> np.ndarray:
        qe = self._query_emb(query)
        return self._emb @ qe.reshape(-1)

    def search(self, query: Query, top_k: int) -> list[int]:
        s = self.scores(query)
        k = min(top_k, len(s))
        idx = np.argpartition(-s, range(k))[:k]
        idx = idx[np.argsort(-s[idx])]
        return [int(i) for i in idx]
