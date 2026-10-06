"""SearchForge · 检索评估指标。

星 (晨星) · 2026-10-07

指标口径统一：
  nDCG@k  使用分级相关性 gain = 2^grade − 1，discount = log2(rank + 1)
  MAP@k   使用二元相关性（grade > 0 为相关）
  MRR@k   首个相关文档排名的倒数
  Recall@k 二元相关性，k 内命中的相关文档占比

无相关文档的查询（num_rel == 0）从聚合中剔除（不填 0，避免"作废伪装成最差"）。
"""

from __future__ import annotations

import math
from typing import Dict, List

from ..core.errors import E500EvalError


def _gains(grade: int) -> float:
    return float((1 << int(grade)) - 1) if grade > 0 else 0.0


def ndcg_at_k(ranking: List[int], grades: Dict[int, int], k: int) -> float:
    if k <= 0:
        raise E500EvalError("k 必须为正")
    if not ranking:
        return 0.0
    dcg = 0.0
    for i, doc in enumerate(ranking[:k]):
        dcg += _gains(grades.get(doc, 0)) / math.log2(i + 2.0)
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = sum(_gains(g) / math.log2(i + 2.0) for i, g in enumerate(ideal))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def map_at_k(ranking: List[int], grades: Dict[int, int], k: int) -> float:
    rel_docs = {d for d, g in grades.items() if g > 0}
    if not rel_docs:
        return 0.0
    hits = 0
    ap = 0.0
    for i, doc in enumerate(ranking[:k]):
        if doc in rel_docs:
            hits += 1
            ap += hits / (i + 1.0)
    return ap / len(rel_docs)


def mrr_at_k(ranking: List[int], grades: Dict[int, int], k: int) -> float:
    rel_docs = {d for d, g in grades.items() if g > 0}
    if not rel_docs:
        return 0.0
    for i, doc in enumerate(ranking[:k]):
        if doc in rel_docs:
            return 1.0 / (i + 1.0)
    return 0.0


def recall_at_k(ranking: List[int], grades: Dict[int, int], k: int) -> float:
    rel_docs = {d for d, g in grades.items() if g > 0}
    if not rel_docs:
        return 0.0
    hits = sum(1 for d in ranking[:k] if d in rel_docs)
    return hits / len(rel_docs)


def evaluate_method(rankings: Dict[int, List[int]], qrels, ks: List[int]) -> Dict[str, float]:
    """在给定 (qid -> ranking) 上计算全部指标，按查询取均值。"""
    out: Dict[str, float] = {}
    skipped = 0
    scores = {f"ndcg@{k}": [] for k in ks}
    scores.update({f"map@{k}": [] for k in ks})
    scores.update({f"mrr@{k}": [] for k in ks})
    scores.update({f"recall@{k}": [] for k in ks})

    for qid, rnk in rankings.items():
        grades = qrels.data.get(qid, {})
        if not any(g > 0 for g in grades.values()):
            skipped += 1
            continue
        for k in ks:
            scores[f"ndcg@{k}"].append(ndcg_at_k(rnk, grades, k))
            scores[f"map@{k}"].append(map_at_k(rnk, grades, k))
            scores[f"mrr@{k}"].append(mrr_at_k(rnk, grades, k))
            scores[f"recall@{k}"].append(recall_at_k(rnk, grades, k))

    for key, vals in scores.items():
        out[key] = float(sum(vals) / len(vals)) if vals else 0.0
    out["_n_queries"] = float(
        sum(1 for qid in rankings if any(g > 0 for g in qrels.data.get(qid, {}).values()))
    )
    out["_n_skipped"] = float(skipped)
    return out
