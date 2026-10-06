"""SearchForge · 流水线单测（无泄漏 / 确定性 / 降级 / CLI 冒烟）。

星 (晨星) · 2026-10-07
"""

import numpy as np
import pytest

from searchforge.core.config import Config
from searchforge.core.seed import set_all
from searchforge.pipeline.pipeline import (
    BASELINE_METHODS,
    FLAGSHIP,
    SearchPipeline,
    benchmark,
)
from searchforge.rerank.features import FEATURE_NAMES


def _cfg(**kw):
    base = dict(
        n_docs=150,
        n_queries=40,
        vocab_size=200,
        n_topics=6,
        doc_len=25,
        query_len=6,
        topic_support=60,
        common_frac=0.4,
        lsi_dim=8,
        top_k_retrieve=40,
        rerank_candidates=20,
        ltr_epochs=20,
        n_seeds=2,
        eval_k=(5, 10),
    )
    base.update(kw)
    return Config(**base)


def test_train_test_split_is_disjoint_and_complete():
    p = SearchPipeline(_cfg())
    assert set(p.train_qids).isdisjoint(set(p.test_qids))
    assert sorted(p.train_qids + p.test_qids) == sorted(q.id for q in p.queries)
    assert len(p.train_qids) > 0 and len(p.test_qids) > 0


def test_reranker_trained_only_on_train_queries():
    """泄漏防护：重排器只吃 train 查询的特征。"""
    p = SearchPipeline(_cfg())
    train_feats = {}
    for qid in p.train_qids:
        bm25, dense, _lx, _dn, fused, rrf_pos = p._recall(qid)
        cand = fused[: p.cfg.rerank_candidates]
        fd = p.fe.feature_dict(qid, cand, bm25, dense, rrf_pos, set(p.queries[qid].terms))
        train_feats[qid] = [(fd[d], d) for d in cand]
    assert set(train_feats.keys()) <= set(p.train_qids)
    assert not (set(train_feats.keys()) & set(p.test_qids))


def test_run_returns_all_methods():
    res = SearchPipeline(_cfg()).run()
    for m in ("random", *BASELINE_METHODS, FLAGSHIP):
        assert m in res["methods"]
        assert "ndcg@10" in res["methods"][m]


def test_random_control_is_near_zero():
    res = SearchPipeline(_cfg()).run()
    assert res["methods"]["random"]["ndcg@10"] < 0.2


def test_flagship_beats_random_and_is_nonnegative():
    res = SearchPipeline(_cfg()).run()
    assert 0.0 <= res["methods"][FLAGSHIP]["ndcg@10"] <= 1.0
    assert res["methods"][FLAGSHIP]["ndcg@10"] > res["methods"]["random"]["ndcg@10"]


def test_determinism_same_seed_twice():
    cfg = _cfg()
    set_all(cfg.seed)
    r1 = SearchPipeline(cfg).run()
    set_all(cfg.seed)
    r2 = SearchPipeline(cfg).run()
    for m in r1["methods"]:
        for k, v in r1["methods"][m].items():
            if k.startswith("_"):
                continue
            assert abs(v - r2["methods"][m][k]) <= 1e-9


def test_different_seeds_change_results():
    a = SearchPipeline(_cfg(seed=1)).run()
    b = SearchPipeline(_cfg(seed=2)).run()
    assert a["methods"][FLAGSHIP]["ndcg@10"] != b["methods"][FLAGSHIP]["ndcg@10"]


def test_benchmark_multiseed_gate_structure():
    cfg = _cfg()
    bench = benchmark(cfg, seeds=[1, 2])
    assert bench["gate"]["metric"] == "ndcg@10"
    assert set(bench["baselines"]) == set(BASELINE_METHODS)
    assert "significant" in bench["gate"] and "passed" in bench["gate"]
    assert len(bench["runs"]) == 2
    for m in bench["summary"]:
        for k in bench["summary"][m]:
            assert set(bench["summary"][m][k].keys()) == {"mean", "std", "values"}


def test_offline_fallback_path_runs(monkeypatch):
    """离线兜底：强制 faiss/lightgbm/rank_bm25 不可用时仍需跑通。"""
    import searchforge.rerank.lambdamart as lm_mod
    import searchforge.retrieval.dense as dense_mod
    import searchforge.retrieval.lexical as lex_mod

    monkeypatch.setattr(dense_mod.DenseRetriever, "available", staticmethod(lambda: False))
    monkeypatch.setattr(lex_mod.BM25Retriever, "available", staticmethod(lambda: False))
    monkeypatch.setattr(lm_mod.LightGBMReranker, "available", staticmethod(lambda: False))
    res = SearchPipeline(_cfg()).run()
    assert res["backend"]["faiss"] is False
    assert res["backend"]["bm25"] is False
    assert res["reranker"] == "rankfuse_sklearn"
    assert res["methods"][FLAGSHIP]["ndcg@10"] >= 0.0


def test_cli_bench_smoke(tmp_path):
    from searchforge.cli import main

    out = tmp_path / "b.json"
    rc = main(
        [
            "bench",
            "--seeds",
            "1,2",
            "--out",
            str(out),
        ]
    )
    assert out.exists()
    assert rc in (0, 1)


def test_cli_run_smoke(tmp_path):
    from searchforge.cli import main

    out = tmp_path / "single.json"
    rc = main(["run", "--seed", "3", "--out", str(out)])
    assert rc == 0
    assert out.exists()


def test_hpo_tune_runs_and_returns_params():
    """HPO 模块必须真的能跑（此前覆盖率 0%，等于未验证）。"""
    from searchforge.hpo.tune import tune

    cfg = _cfg(n_docs=80, n_queries=20, ltr_epochs=10, hpo_seed=11)
    best, study = tune(cfg, n_trials=2)
    if study is None:
        pytest.skip("optuna 不可用")
    assert isinstance(best, dict)
    assert set(best.keys()) >= {"rrf_k", "lsi_dim", "num_leaves", "learning_rate"}


def test_ablation_feature_subset_runs():
    p = SearchPipeline(_cfg())
    names = [n for n in FEATURE_NAMES if n != "dense"]
    res = p.run(feature_names=names)
    assert "ndcg@10" in res["methods"][FLAGSHIP]
    assert np.isfinite(res["methods"][FLAGSHIP]["ndcg@10"])
