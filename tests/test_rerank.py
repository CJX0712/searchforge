"""SearchForge · 重排层单测（特征 / LambdaMART / 离线兜底）。

星 (晨星) · 2026-10-07
"""

import pytest

from searchforge.core.config import Config
from searchforge.core.errors import E400RerankError
from searchforge.data.generator import SyntheticDataset
from searchforge.rerank import (
    FEATURE_NAMES,
    LightGBMReranker,
    SklearnRanker,
    build_reranker,
)
from searchforge.rerank.features import FeatureExtractor
from searchforge.retrieval import BM25Retriever, DenseRetriever


def _pipe_parts(**kw):
    base = dict(
        n_docs=100,
        n_queries=20,
        vocab_size=150,
        n_topics=5,
        doc_len=25,
        query_len=6,
        topic_support=50,
        common_frac=0.3,
        lsi_dim=8,
        top_k_retrieve=40,
        rerank_candidates=20,
        ltr_epochs=20,
    )
    base.update(kw)
    cfg = Config(**base)
    ds = SyntheticDataset(cfg)
    lex = BM25Retriever().fit(ds.corpus)
    den = DenseRetriever(lsi_dim=cfg.lsi_dim, random_state=cfg.seed).fit(ds.corpus)
    fe = FeatureExtractor(ds.corpus, lex, den, cfg.rrf_k)
    return cfg, ds, lex, den, fe


def test_feature_extractor_produces_all_features():
    cfg, ds, lex, den, fe = _pipe_parts()
    q = ds.queries[0]
    bm25 = lex.scores(q)
    dense = den.scores(q)
    fused = list(range(10))
    rrf_pos = {d: i for i, d in enumerate(fused)}
    out = fe.feature_dict(q.id, fused, bm25, dense, rrf_pos, set(q.terms))
    assert len(out) == 10
    for f in out.values():
        assert set(f.keys()) == set(FEATURE_NAMES)


def test_feature_extractor_rrf_fallback_for_unseen_doc():
    cfg, ds, lex, den, fe = _pipe_parts()
    q = ds.queries[0]
    bm25 = lex.scores(q)
    dense = den.scores(q)
    out = fe.feature_dict(q.id, [7], bm25, dense, {}, set(q.terms))
    assert out[7]["rrf"] > 0.0


def test_lambdamart_fit_and_rerank():
    cfg, ds, lex, den, fe = _pipe_parts()
    if not LightGBMReranker.available():
        pytest.skip("lightgbm 不可用")
    train = [q.id for q in ds.queries[:10]]
    feat_by_qid = {}
    for qid in train:
        q = ds.queries[qid]
        bm25 = lex.scores(q)
        dense = den.scores(q)
        fused = list(range(20))
        rrf_pos = {d: i for i, d in enumerate(fused)}
        fd = fe.feature_dict(qid, fused, bm25, dense, rrf_pos, set(q.terms))
        feat_by_qid[qid] = [(fd[d], d) for d in fused]
    rr = LightGBMReranker(random_state=0, n_estimators=20).fit(
        ds.corpus, ds.qrels, train, feat_by_qid
    )
    q = ds.queries[0]
    bm25 = lex.scores(q)
    dense = den.scores(q)
    fused = list(range(20))
    rrf_pos = {d: i for i, d in enumerate(fused)}
    fd = fe.feature_dict(q.id, fused, bm25, dense, rrf_pos, set(q.terms))
    out = rr.rerank(q.id, fd)
    assert sorted(out) == sorted(fd.keys())


def test_reranker_guard_unfitted():
    rr = LightGBMReranker()
    with pytest.raises(E400RerankError):
        rr.rerank(0, {1: {n: 0.0 for n in FEATURE_NAMES}})


def test_reranker_guard_no_positive_labels():
    cfg, ds, lex, den, fe = _pipe_parts()
    from searchforge.core.types import Qrels

    empty = Qrels(data={q.id: {} for q in ds.queries})
    train = [q.id for q in ds.queries[:10]]
    feat_by_qid = {qid: [] for qid in train}
    with pytest.raises(E400RerankError):
        LightGBMReranker().fit(ds.corpus, empty, train, feat_by_qid)


def test_sklearn_fallback_rerank():
    cfg, ds, lex, den, fe = _pipe_parts()
    train = [q.id for q in ds.queries[:10]]
    feat_by_qid = {}
    for qid in train:
        q = ds.queries[qid]
        bm25 = lex.scores(q)
        dense = den.scores(q)
        fused = list(range(20))
        rrf_pos = {d: i for i, d in enumerate(fused)}
        fd = fe.feature_dict(qid, fused, bm25, dense, rrf_pos, set(q.terms))
        feat_by_qid[qid] = [(fd[d], d) for d in fused]
    rr = SklearnRanker(random_state=0).fit(ds.corpus, ds.qrels, train, feat_by_qid)
    q = ds.queries[0]
    bm25 = lex.scores(q)
    dense = den.scores(q)
    fused = list(range(20))
    rrf_pos = {d: i for i, d in enumerate(fused)}
    fd = fe.feature_dict(q.id, fused, bm25, dense, rrf_pos, set(q.terms))
    assert sorted(rr.rerank(q.id, fd)) == sorted(fd.keys())


def test_build_reranker_prefers_lightgbm_when_available():
    rr = build_reranker(random_state=0)
    if LightGBMReranker.available():
        assert rr.name == "rankfuse_lambdamart"
    else:
        assert rr.name == "rankfuse_sklearn"


def test_feature_subset_is_respected():
    """消融用特征子集：传入子集后模型只应看到子集列。"""
    cfg, ds, lex, den, fe = _pipe_parts()
    if not LightGBMReranker.available():
        pytest.skip("lightgbm 不可用")
    names = [n for n in FEATURE_NAMES if n != "dense"]
    train = [q.id for q in ds.queries[:10]]
    feat_by_qid = {}
    for qid in train:
        q = ds.queries[qid]
        bm25 = lex.scores(q)
        dense = den.scores(q)
        fused = list(range(20))
        rrf_pos = {d: i for i, d in enumerate(fused)}
        fd = fe.feature_dict(qid, fused, bm25, dense, rrf_pos, set(q.terms))
        feat_by_qid[qid] = [(fd[d], d) for d in fused]
    rr = LightGBMReranker(random_state=0, n_estimators=10, feature_names=names).fit(
        ds.corpus, ds.qrels, train, feat_by_qid
    )
    assert rr.feature_names == names
    assert rr._model.n_features_in_ == len(names)
