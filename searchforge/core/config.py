"""SearchForge · 全局配置（支持 ENV_SEARCHFORGE_* 覆盖 + 简单 schema 校验）。

星 (晨星) · 2026-10-07
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

PREFIX = "SEARCHFORGE_"


@dataclass
class Config:
    """系统可调参数。所有字段均可被环境变量 ``SEARCHFORGE_<NAME>`` 覆盖。"""

    # ---- 随机性与复现 ----
    seed: int = 42

    # ---- 合成语料 DGP（难度旋钮经扫描定在甜点，见 docs/architecture.md §5）----
    n_docs: int = 1200
    n_queries: int = 300
    vocab_size: int = 1500
    n_topics: int = 20
    doc_len: int = 60
    query_len: int = 10
    topic_concentration: float = 1.0  # Dirichlet 浓度（1.0 = 支撑集上近似均匀）
    topic_support: int = 200  # 每个主题的词支撑集大小：越小→同主题重叠越多、越易
    common_frac: float = 0.50  # 支撑集中跨主题共享词比例：越大→词法诱导负例越多
    query_noise: float = 0.30  # 查询词混入背景噪声的比例
    overlap_threshold: int = 3  # 同主题且共享词数 >= 此值 → grade 2
    require_topic: bool = True  # 相关性要求同主题（使词法信号成为必要但不充分条件）
    strong_offset: int = 2  # grade=3 档相比 grade=2 档的共享词数增量

    # ---- 检索 / 融合 ----
    lsi_dim: int = 32  # HPO(Optuna, seed=9042) 选中；跨 seed 稳健（见 docs/architecture.md §5）
    rrf_k: int = 60  # Cormack et al. 2009 经典值；实测与 HPO 给出的 118 无显著差异
    top_k_retrieve: int = 120  # 初排召回候选数
    eval_k: Tuple[int, ...] = (5, 10, 20, 100)

    # ---- 学习重排 LTR ----
    train_ratio: float = 0.6
    rerank_candidates: int = 60  # 进入重排阶段的候选数
    ltr_epochs: int = 100

    # ---- 基准 / 验收 ----
    n_seeds: int = 3
    hpo_seed: int = 9042  # HPO 用独立种子，与 benchmark 种子不相交（防泄漏）
    # S 级门槛：旗舰 nDCG@10 相对最强基线提升 >= 此相对值且统计显著。
    # 0.05 是**开工前**按 pilot 实测（+11.5%，std 0.018）定死的结构性余量，
    # 不为结果回头迁就（pilot 值 11.5% 是门槛的 2.3 倍）。
    ndcg_gain_rel: float = 0.05

    def __post_init__(self) -> None:
        if self.n_docs <= 0 or self.n_queries <= 0 or self.vocab_size <= 0:
            from .errors import E100ConfigError

            raise E100ConfigError("n_docs/n_queries/vocab_size 必须为正")
        if not (0.0 < self.train_ratio < 1.0):
            from .errors import E100ConfigError

            raise E100ConfigError("train_ratio 必须介于 (0,1)")
        if self.lsi_dim > self.vocab_size:
            from .errors import E100ConfigError

            raise E100ConfigError("lsi_dim 不能超过 vocab_size")

    @classmethod
    def from_env(cls) -> "Config":
        """读取 ``SEARCHFORGE_*`` 环境变量覆盖默认配置。"""
        overrides: Dict[str, str] = {}
        for key in os.environ:
            if key.startswith(PREFIX):
                overrides[key[len(PREFIX) :].lower()] = os.environ[key]
        if not overrides:
            return cls()
        # 类型化转换
        ints = {
            "seed",
            "n_docs",
            "n_queries",
            "vocab_size",
            "n_topics",
            "doc_len",
            "query_len",
            "lsi_dim",
            "rrf_k",
            "top_k_retrieve",
            "rerank_candidates",
            "ltr_epochs",
            "n_seeds",
            "hpo_seed",
            "overlap_threshold",
            "strong_offset",
            "topic_support",
        }
        floats = {"topic_concentration", "query_noise", "train_ratio", "ndcg_gain_rel"}
        bools = {"require_topic"}
        cfg = cls()
        for name, val in overrides.items():
            if name == "eval_k":
                cfg.eval_k = tuple(int(x) for x in val.split(",") if x.strip())
                continue
            if hasattr(cfg, name):
                if name in ints:
                    setattr(cfg, name, int(val))
                elif name in floats:
                    setattr(cfg, name, float(val))
                elif name in bools:
                    setattr(cfg, name, str(val).strip().lower() in ("1", "true", "yes"))
                else:
                    setattr(cfg, name, val)
        cfg.__post_init__()
        return cfg

    def as_dict(self) -> Dict:
        return {
            "seed": self.seed,
            "n_docs": self.n_docs,
            "n_queries": self.n_queries,
            "vocab_size": self.vocab_size,
            "n_topics": self.n_topics,
            "doc_len": self.doc_len,
            "query_len": self.query_len,
            "topic_concentration": self.topic_concentration,
            "topic_support": self.topic_support,
            "common_frac": self.common_frac,
            "query_noise": self.query_noise,
            "overlap_threshold": self.overlap_threshold,
            "require_topic": self.require_topic,
            "strong_offset": self.strong_offset,
            "lsi_dim": self.lsi_dim,
            "rrf_k": self.rrf_k,
            "top_k_retrieve": self.top_k_retrieve,
            "eval_k": list(self.eval_k),
            "train_ratio": self.train_ratio,
            "rerank_candidates": self.rerank_candidates,
            "ltr_epochs": self.ltr_epochs,
            "n_seeds": self.n_seeds,
            "hpo_seed": self.hpo_seed,
            "ndcg_gain_rel": self.ndcg_gain_rel,
        }
