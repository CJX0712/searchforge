"""SearchForge · 词法检索（BM25，顶级开源 rank_bm25；缺失则 TF-IDF 余弦兜底）。

星 (晨星) · 2026-10-07
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E300RetrievalError
from ..core.interfaces import Retriever
from ..core.types import Corpus, Query


def _tokens(doc_terms) -> list[str]:
    return [str(int(t)) for t in doc_terms]


class BM25Retriever:
    """基于 rank_bm25 的 BM25Okapi 词法检索器（工业级基线）。"""

    name = "bm25"

    def __init__(self) -> None:
        self._index = None
        self._corpus = None

    @staticmethod
    def available() -> bool:
        try:
            import rank_bm25  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, corpus: Corpus) -> "BM25Retriever":
        if not self.available():
            raise E300RetrievalError("rank_bm25 不可用")
        from rank_bm25 import BM25Okapi

        self._corpus = corpus
        corpus_tokens = [_tokens(d.terms) for d in corpus.docs]
        self._index = BM25Okapi(corpus_tokens)
        return self

    def scores(self, query: Query) -> np.ndarray:
        """返回 [n_docs] 的 BM25 分数。"""
        if self._index is None:
            raise E300RetrievalError("BM25Retriever 未 fit")
        return np.asarray(self._index.get_scores(_tokens(query.terms)), dtype=np.float64)

    def search(self, query: Query, top_k: int) -> list[int]:
        s = self.scores(query)
        k = min(top_k, len(s))
        idx = np.argpartition(-s, range(k))[:k]
        idx = idx[np.argsort(-s[idx])]
        return [int(i) for i in idx]


class TfidfCosineRetriever:
    """离线兜底：TF-IDF 余弦相似度（无 rank_bm25 时）。"""

    name = "tfidf_cosine"

    def __init__(self, vocab_size: int) -> None:
        from ._tfidf import TfidfIndex

        self._tfidf = TfidfIndex(vocab_size)
        self._doc_vecs = None

    @staticmethod
    def available() -> bool:
        return True

    def fit(self, corpus: Corpus) -> "TfidfCosineRetriever":
        self._doc_vecs = self._tfidf.fit(corpus).transform_docs(corpus)
        return self

    def scores(self, query: Query) -> np.ndarray:
        if self._doc_vecs is None:
            raise E300RetrievalError("TfidfCosineRetriever 未 fit")
        qv = self._tfidf.transform_queries([query])  # [1, V]，已用 fit 的 idf
        return self._doc_vecs @ qv[0]

    def search(self, query: Query, top_k: int) -> list[int]:
        s = self.scores(query)
        k = min(top_k, len(s))
        idx = np.argpartition(-s, range(k))[:k]
        idx = idx[np.argsort(-s[idx])]
        return [int(i) for i in idx]


def LexicalRetriever(vocab_size: int) -> Retriever:
    """工厂：优先 BM25，否则 TF-IDF 余弦兜底。"""
    if BM25Retriever.available():
        return BM25Retriever()
    return TfidfCosineRetriever(vocab_size)
