"""SearchForge · 混合召回融合（Reciprocal Rank Fusion, Cormack et al. 2009）。

星 (晨星) · 2026-10-07

RRF 是无需训练的两路召回融合方法，被工业级搜索引擎（含 RAG 检索）广泛采用。
score(d) = Σ_{r ∈ 召回源} 1 / (k + rank_r(d))，按分数降序输出。
"""

from __future__ import annotations


class RRFFusion:
    name = "rrf"

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(self, rankings: dict[str, list[int]], top_k: int) -> list[int]:
        if not rankings:
            return []
        score = {}
        for _src, rnk in rankings.items():
            for rank, doc_id in enumerate(rnk):
                score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (self.k + rank)
        # 并列时按 doc_id 升序打破平局 ⟹ 结果对「召回路传入顺序」置换不变
        # （否则 sorted 的稳定性会让输出依赖源字典的插入顺序）
        ordered = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
        return [int(d) for d, _ in ordered[:top_k]]
