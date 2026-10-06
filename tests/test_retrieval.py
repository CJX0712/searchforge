"""SearchForge · 检索层单测（含 FAISS/离线兜底等价性这一硬不变量）。

星 (晨星) · 2026-10-07
"""

import numpy as np
import pytest

from searchforge.core.config import Config
from searchforge.data.generator import SyntheticDataset
from searchforge.retrieval import (
    BM25Retriever,
    DenseRetriever,
    NumpyCosineRetriever,
    RRFFusion,
    TfidfCosineRetriever,
)


def _ds(**kw):
    base = dict(
        n_docs=100,
        n_queries=10,
        vocab_size=150,
        n_topics=5,
        doc_len=25,
        query_len=6,
        topic_support=50,
        common_frac=0.3,
    )
    base.update(kw)
    return SyntheticDataset(Config(**base))


def test_bm25_search_returns_topk_ordered():
    ds = _ds()
    r = BM25Retriever()
    if not r.available():
        pytest.skip("rank_bm25 不可用")
    r.fit(ds.corpus)
    res = r.search(ds.queries[0], 10)
    assert len(res) == 10
    s = r.scores(ds.queries[0])
    vals = [s[i] for i in res]
    assert vals == sorted(vals, reverse=True)


def test_tfidf_fallback_works():
    ds = _ds()
    r = TfidfCosineRetriever(vocab_size=150)
    r.fit(ds.corpus)
    res = r.search(ds.queries[0], 5)
    assert len(res) == 5


def test_faiss_and_numpy_fallback_are_equivalent():
    """硬不变量：FAISS(IndexFlatIP) 与 numpy 精确余弦的 top-k 排序必须一致。"""
    ds = _ds()
    faiss_r = DenseRetriever(lsi_dim=8, random_state=0).fit(ds.corpus)
    np_r = NumpyCosineRetriever(lsi_dim=8, random_state=0).fit(ds.corpus)
    if not DenseRetriever.available():
        pytest.skip("faiss 不可用")
    for q in ds.queries[:5]:
        assert faiss_r.search(q, 10) == np_r.search(q, 10)


def test_dense_scores_shape():
    ds = _ds()
    r = DenseRetriever(lsi_dim=8).fit(ds.corpus)
    s = r.scores(ds.queries[0])
    assert s.shape == (ds.corpus.n_docs,)


def test_dense_numpy_fallback_scores_shape():
    ds = _ds()
    r = NumpyCosineRetriever(lsi_dim=8).fit(ds.corpus)
    s = r.scores(ds.queries[0])
    assert s.shape == (ds.corpus.n_docs,)


def test_rrf_favours_docs_ranked_high_in_both():
    """RRF：两路都排首位的文档必须排第一。"""
    f = RRFFusion(k=60)
    out = f.fuse({"a": [1, 2, 3], "b": [1, 3, 2]}, 10)
    assert out[0] == 1
    # doc1 得分 = 2/60，doc2/doc3 = 1/60 + 1/61，故 doc2 与 doc3 居中
    assert out[-1] != 1


def test_rrf_handles_empty():
    assert RRFFusion().fuse({}, 10) == []
    assert RRFFusion().fuse({"a": []}, 10) == []


def test_rrf_respects_topk():
    out = RRFFusion().fuse({"a": list(range(50)), "b": list(range(49, -1, -1))}, 7)
    assert len(out) == 7


def test_rrf_tie_break_is_deterministic():
    f = RRFFusion(k=60)
    a = f.fuse({"a": [5, 6], "b": [6, 5]}, 10)
    b = f.fuse({"b": [6, 5], "a": [5, 6]}, 10)
    assert a == b
