"""SearchForge · 接口契约（Protocol）。

星 (晨星) · 2026-10-07

调用方向单向无环：cli → pipeline → {data, retrieval, rerank, eval} → core。
所有模块实现下列 Protocol 即可独立验证。
"""

from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable

from .types import Corpus, Qrels, Query


@runtime_checkable
class Retriever(Protocol):
    """词法 / 稠密检索器：在语料上建索引，给定查询返回排序候选。"""

    name: str

    def fit(self, corpus: Corpus) -> "Retriever": ...

    def search(self, query: Query, top_k: int) -> List[int]:
        """返回 doc_id 列表（按相关性降序）。"""
        ...

    def available(self) -> bool: ...


@runtime_checkable
class Fusion(Protocol):
    """多路召回结果的融合策略（如 RRF）。"""

    name: str

    def fuse(self, rankings: Dict[str, List[int]], top_k: int) -> List[int]: ...


@runtime_checkable
class Reranker(Protocol):
    """学习重排器：基于候选特征对初排列表重排序。"""

    name: str

    def fit(
        self,
        corpus: Corpus,
        qrels: Qrels,
        train_qids: List[int],
        features: Dict[int, Dict[int, Dict[str, float]]],
    ) -> "Reranker": ...

    def rerank(self, qid: int, candidates: Dict[int, Dict[str, float]]) -> List[int]:
        """输入候选 doc_id -> 特征，返回重排后的 doc_id 列表。"""
        ...
