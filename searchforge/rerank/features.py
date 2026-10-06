"""SearchForge · 重排特征提取。

星 (晨星) · 2026-10-07

每个候选 (query, doc) 抽取的特征：
  bm25        词法检索原始分
  dense       稠密检索余弦相似度
  rrf         在 RRF 融合列表中的排名倒数（1/(k+rank)）
  overlap     查询与文档共享词数（带重数）
  doc_len     log(1 + 文档长度)
  query_idf   查询词 idf 之和（查询特异性代理，越大越难）

所有特征仅在「已见文档集合」上计算，无任何来自测试标签的信息 ⟹ 无泄漏。
"""

from __future__ import annotations

import numpy as np

from ..core.types import Corpus, Query
from ..retrieval.hybrid import RRFFusion

FEATURE_NAMES = ["bm25", "dense", "rrf", "overlap", "doc_len", "query_idf"]


class FeatureExtractor:
    def __init__(self, corpus: Corpus, lexical, dense, rrf_k: int = 60) -> None:
        self.corpus = corpus
        self.lexical = lexical
        self.dense = dense
        self.rrf_k = rrf_k
        self._doc_sets = [set(d.terms) for d in corpus.docs]
        # 预计算 df / idf
        V = int(max((max(d.terms) for d in corpus.docs if d.terms), default=0)) + 1
        df = np.zeros(V, dtype=np.float64)
        for d in corpus.docs:
            for t in set(d.terms):
                df[t] += 1.0
        self._idf = np.log((1.0 + len(corpus.docs)) / (1.0 + df)) + 1.0
        self._V = V

    def per_query(self, query: Query, top_k: int):
        """返回 (bm25[N], dense[N], rrf_pos: dict, qset)。"""
        bm25 = self.lexical.scores(query)
        dense = self.dense.scores(query)
        rk = {
            self.lexical.name: self.lexical.search(query, top_k),
            self.dense.name: self.dense.search(query, top_k),
        }
        fused = RRFFusion(self.rrf_k).fuse(rk, top_k)
        rrf_pos = {d: i for i, d in enumerate(fused)}
        qset = set(query.terms)
        return bm25, dense, rrf_pos, qset

    def query_idf(self, qset: set) -> float:
        s = 0.0
        for t in qset:
            if 0 <= t < self._V:
                s += float(self._idf[t])
        return s

    def feature_dict(self, qid: int, cand_ids, bm25, dense, rrf_pos, qset):
        out = {}
        n = len(rrf_pos)
        for d in cand_ids:
            o = len(qset & self._doc_sets[d])
            out[d] = {
                "bm25": float(bm25[d]),
                "dense": float(dense[d]),
                "rrf": float(1.0 / (self.rrf_k + rrf_pos.get(d, n))),
                "overlap": float(o),
                "doc_len": float(np.log1p(len(self.corpus.docs[d].terms))),
                "query_idf": float(self.query_idf(qset)),
            }
        return out
