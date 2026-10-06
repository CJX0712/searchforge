"""SearchForge · 评估指标单测（手算对照，防静默错误）。

星 (晨星) · 2026-10-07
"""

import math

import pytest

from searchforge.core.types import Qrels
from searchforge.eval.metrics import (
    evaluate_method,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

# grade: a=3, b=2, c=1（三篇均相关，用于 nDCG 分级测试）
GRADES = {"a": 3, "b": 2, "c": 1}
# 二值标注：只有 a、b 相关（用于 MAP / MRR / Recall 的手算对照）
BINARY = {"a": 1, "b": 1, "x": 0, "y": 0}


def gain(g):
    return float((1 << g) - 1)


def test_ndcg_perfect_ordering():
    # 理想顺序 a,b → DCG == IDCG ⇒ 1.0
    assert ndcg_at_k(["a", "b", "c"], GRADES, 2) == pytest.approx(1.0, abs=1e-12)


def test_ndcg_reversed_ordering_matches_handcalc():
    # 逆序 c,b → DCG = gain(1)/log2(2) + gain(2)/log2(3)
    dcg = gain(1) / math.log2(2) + gain(2) / math.log2(3)
    idcg = gain(3) / math.log2(2) + gain(2) / math.log2(3)
    assert ndcg_at_k(["c", "b", "a"], GRADES, 2) == pytest.approx(dcg / idcg, abs=1e-12)


def test_ndcg_no_relevant_in_ranking():
    assert ndcg_at_k(["x", "y"], GRADES, 2) == 0.0


def test_ndcg_empty_ranking():
    assert ndcg_at_k([], GRADES, 10) == 0.0


def test_map_handcalc():
    # rel = {a,b}；ranking = [x,a,y,b]，命中在 rank2 与 rank4
    ranks = ["x", "a", "y", "b"]
    ap = 1 / 2 + 2 / 4  # = 1.0
    assert map_at_k(ranks, BINARY, 4) == pytest.approx(ap / 2, abs=1e-12)


def test_mrr_and_recall():
    ranks = ["x", "a", "y", "b"]
    assert mrr_at_k(ranks, BINARY, 4) == pytest.approx(0.5, abs=1e-12)
    assert recall_at_k(ranks, BINARY, 4) == pytest.approx(1.0, abs=1e-12)
    assert recall_at_k(ranks, BINARY, 1) == pytest.approx(0.0, abs=1e-12)
    # 截断到 k=2 时只召回 1/2
    assert recall_at_k(ranks, BINARY, 2) == pytest.approx(0.5, abs=1e-12)


def test_metrics_zero_when_no_relevant():
    empty = {"z": 0}
    assert map_at_k(["z"], empty, 5) == 0.0
    assert mrr_at_k(["z"], empty, 5) == 0.0
    assert recall_at_k(["z"], empty, 5) == 0.0


def test_evaluate_method_skips_queries_without_relevant():
    qrels = Qrels(data={1: {"a": 2, "b": 1}, 2: {"z": 0}})
    rankings = {1: ["a", "b", "c"], 2: ["z"]}
    out = evaluate_method(rankings, qrels, [5, 10])
    # 只有 qid=1 参与聚合；qid=2（无相关文档）被剔除而不是填 0
    assert out["_n_queries"] == 1.0
    assert out["_n_skipped"] == 1.0
    assert out["ndcg@5"] == pytest.approx(1.0, abs=1e-12)
