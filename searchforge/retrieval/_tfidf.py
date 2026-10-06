"""SearchForge · 共享 TF-IDF 工具（稠密 LSI 与词法兜底共用，确定性）。

星 (晨星) · 2026-10-07
"""

from __future__ import annotations

import numpy as np

from ..core.types import Corpus


def _term_counts(docs_terms, vocab_size: int) -> np.ndarray:
    n = len(docs_terms)
    mat = np.zeros((n, vocab_size), dtype=np.float64)
    for i, terms in enumerate(docs_terms):
        if not terms:
            continue
        t = np.asarray(terms, dtype=np.int64)
        uniq, cnt = np.unique(t, return_counts=True)
        mat[i, uniq] = cnt
    return mat


class TfidfIndex:
    """在语料上拟合并转换文档/查询为 TF-IDF 向量（L2 归一化）。"""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.idf = None
        self._n = 0

    def fit(self, corpus: Corpus) -> "TfidfIndex":
        docs_terms = [d.terms for d in corpus.docs]
        counts = _term_counts(docs_terms, self.vocab_size)
        self._n = counts.shape[0]
        df = (counts > 0).sum(axis=0)  # [V]
        # sklearn 平滑 idf
        self.idf = np.log((1.0 + self._n) / (1.0 + df)) + 1.0
        return self

    def _transform(self, terms_list) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("TfidfIndex 未 fit")
        counts = _term_counts(terms_list, self.vocab_size)
        tfidf = counts * self.idf  # [m, V]
        # L2 归一化（零向量保持零向量）
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tfidf = tfidf / norms
        return tfidf

    def transform_docs(self, corpus: Corpus) -> np.ndarray:
        return self._transform([d.terms for d in corpus.docs])

    def transform_queries(self, queries) -> np.ndarray:
        return self._transform([q.terms for q in queries])
