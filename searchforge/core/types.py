"""SearchForge · 领域类型定义（dataclass）。

星 (晨星) · 2026-10-07
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Document:
    """一篇文档。terms 为词表索引组成的 token 序列。"""

    id: int
    terms: List[int]
    topic: int


@dataclass
class Query:
    """一条查询。source 为生成该查询时抽样自的相关文档 id。"""

    id: int
    terms: List[int]
    topic: int
    source: int


@dataclass
class Corpus:
    """文档集合 + 生成时使用的潜在主题-词分布（仅供可复现与诊断）。"""

    docs: List[Document]
    topic_word: "object"  # np.ndarray [K, V]
    doc_topic: "object"  # np.ndarray [N]

    @property
    def n_docs(self) -> int:
        return len(self.docs)


@dataclass
class Qrels:
    """相关性标注：qid -> {doc_id: grade}。grade 为整数（0=不相关）。"""

    data: Dict[int, Dict[int, int]] = field(default_factory=dict)

    def relevant(self, qid: int) -> List[int]:
        return [d for d, g in self.data.get(qid, {}).items() if g > 0]

    def grade(self, qid: int, doc_id: int) -> int:
        return self.data.get(qid, {}).get(doc_id, 0)


@dataclass
class Candidate:
    """单个候选文档（含各检索器原始分，供重排特征使用）。"""

    doc_id: int
    score: float  # 融合后的启发式分（仅用于初排/对照）
    bm25: float = 0.0
    dense: float = 0.0
    rrf: float = 0.0
    overlap: int = 0
    doc_len: int = 0


@dataclass
class RetrievalResult:
    """某检索器对某 query 的排序结果：doc_id 列表（已按分数降序）。"""

    qid: int
    ranking: List[int]
    scores: Dict[int, float] = field(default_factory=dict)
