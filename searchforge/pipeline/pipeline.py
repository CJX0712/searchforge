"""SearchForge · 端到端检索流水线与多 seed 基准。

星 (晨星) · 2026-10-07

流程：合成语料 → BM25/LSI-FAISS 双路召回 → RRF 融合初排 → LambdaMART 学习重排 → 评估。
查询按 train/test 划分：重排器**只**在 train 查询上训练，在 test 查询上评估（无泄漏）。
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from ..core.errors import E400RerankError
from ..core.seed import set_all
from ..data.generator import SyntheticDataset
from ..eval.metrics import evaluate_method
from ..rerank import build_reranker
from ..rerank.features import FEATURE_NAMES, FeatureExtractor
from ..retrieval import (
    BM25Retriever,
    DenseRetriever,
    LexicalRetriever,
    NumpyCosineRetriever,
    RRFFusion,
)

BASELINE_METHODS = ("lexical", "dense", "hybrid")
FLAGSHIP = "rankfuse"


def _topk(scores: np.ndarray, k: int) -> list[int]:
    scores = np.asarray(scores, dtype=np.float64)
    k = min(k, len(scores))
    if k <= 0:
        return []
    idx = np.argsort(-scores, kind="stable")[:k]
    return [int(i) for i in idx]


class SearchPipeline:
    def __init__(self, cfg, dataset: SyntheticDataset | None = None) -> None:
        self.cfg = cfg
        set_all(cfg.seed)
        self.dataset = dataset if dataset is not None else SyntheticDataset(cfg)
        self.corpus = self.dataset.corpus
        self.queries = self.dataset.queries
        self.qrels = self.dataset.qrels

        self.lexical = LexicalRetriever(cfg.vocab_size)
        self.lexical.fit(self.corpus)
        if DenseRetriever.available():
            self.dense = DenseRetriever(cfg.lsi_dim, random_state=cfg.seed)
        else:
            self.dense = NumpyCosineRetriever(cfg.lsi_dim, random_state=cfg.seed)
        self.dense.fit(self.corpus)
        self.fe = FeatureExtractor(self.corpus, self.lexical, self.dense, cfg.rrf_k)
        self._split()

    def _split(self) -> None:
        rng = np.random.default_rng(self.cfg.seed + 12345)
        idx = rng.permutation(len(self.queries))
        n_train = int(len(idx) * self.cfg.train_ratio)
        self.train_qids = sorted(int(self.queries[i].id) for i in idx[:n_train])
        self.test_qids = sorted(int(self.queries[i].id) for i in idx[n_train:])

    def _recall(self, qid: int):
        q = self.queries[qid]
        bm25 = np.asarray(self.lexical.scores(q), dtype=np.float64)
        dense = np.asarray(self.dense.scores(q), dtype=np.float64)
        lex_r = _topk(bm25, self.cfg.top_k_retrieve)
        den_r = self.dense.search(q, self.cfg.top_k_retrieve)  # 走 FAISS（可用时）
        fused = RRFFusion(self.cfg.rrf_k).fuse(
            {"lexical": lex_r, "dense": den_r}, self.cfg.top_k_retrieve
        )
        rrf_pos = {d: i for i, d in enumerate(fused)}
        return bm25, dense, lex_r, den_r, fused, rrf_pos

    def _train_reranker(self, rerank_params=None, feature_names=None):
        train_feats = {}
        for qid in self.train_qids:
            bm25, dense, _lx, _dn, fused, rrf_pos = self._recall(qid)
            cand = fused[: self.cfg.rerank_candidates]
            qset = set(self.queries[qid].terms)
            fd = self.fe.feature_dict(qid, cand, bm25, dense, rrf_pos, qset)
            train_feats[qid] = [(fd[d], d) for d in cand]
        params = dict(rerank_params or {})
        params["feature_names"] = feature_names or list(FEATURE_NAMES)
        rr = build_reranker(random_state=self.cfg.seed, **params)
        try:
            rr.fit(self.corpus, self.qrels, self.train_qids, train_feats)
        except E400RerankError:
            return None
        return rr

    def run(self, rerank_params=None, feature_names=None) -> dict:
        t0 = time.time()
        reranker = self._train_reranker(rerank_params, feature_names)

        rankings: dict[str, dict[int, list[int]]] = {
            "random": {},
            "lexical": {},
            "dense": {},
            "hybrid": {},
            FLAGSHIP: {},
        }
        for qid in self.test_qids:
            bm25, dense, lex_r, den_r, fused, rrf_pos = self._recall(qid)
            rankings["lexical"][qid] = lex_r
            rankings["dense"][qid] = den_r
            rankings["hybrid"][qid] = fused
            rng = np.random.default_rng(self.cfg.seed * 1000 + qid)
            rankings["random"][qid] = [
                int(x) for x in rng.permutation(self.corpus.n_docs)[: self.cfg.top_k_retrieve]
            ]
            if reranker is None:
                rankings[FLAGSHIP][qid] = fused
            else:
                cand = fused[: self.cfg.rerank_candidates]
                qset = set(self.queries[qid].terms)
                fd = self.fe.feature_dict(qid, cand, bm25, dense, rrf_pos, qset)
                ordered = reranker.rerank(qid, fd)
                seen = set(ordered)
                rest = [d for d in fused if d not in seen]
                rankings[FLAGSHIP][qid] = ordered + rest

        ks = list(self.cfg.eval_k)
        methods = {m: evaluate_method(rankings[m], self.qrels, ks) for m in rankings}
        return {
            "methods": methods,
            "diagnosis": dict(self.dataset.diagnosis),
            "config": self.cfg.as_dict(),
            "elapsed_sec": float(time.time() - t0),
            "n_train": len(self.train_qids),
            "n_test": len(self.test_qids),
            "reranker": getattr(reranker, "name", None),
            "backend": {
                "lexical": self.lexical.name,
                "dense": self.dense.name,
                "faiss": bool(DenseRetriever.available()),
                "bm25": bool(BM25Retriever.available()),
            },
        }


def benchmark(cfg, seeds: list[int] | None = None) -> dict:
    """多 seed 跑基准，返回逐 seed 结果 + mean±std 汇总 + 显著性判定。"""
    seeds = seeds or list(range(cfg.seed, cfg.seed + cfg.n_seeds))
    runs = []
    for s in seeds:
        scfg = replace(cfg, seed=int(s))
        pipe = SearchPipeline(scfg)
        res = pipe.run()
        runs.append({"seed": int(s), **res})

    metric_keys = sorted(runs[0]["methods"][FLAGSHIP].keys())
    summary = {}
    for m in runs[0]["methods"]:
        summary[m] = {}
        for k in metric_keys:
            vals = np.asarray([r["methods"][m][k] for r in runs], dtype=np.float64)
            summary[m][k] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
                "values": [float(v) for v in vals],
            }

    # 每个 seed 取基线最大值作为「最强基线」；再跨 seed 汇总
    per_seed_strong = []
    for r in runs:
        per_seed_strong.append(max(r["methods"][b]["ndcg@10"] for b in BASELINE_METHODS))
    strong = np.asarray(per_seed_strong, dtype=np.float64)
    flag = np.asarray([r["methods"][FLAGSHIP]["ndcg@10"] for r in runs], dtype=np.float64)
    diff = float(flag.mean() - strong.mean())
    sig = diff > 0.5 * (float(flag.std(ddof=0)) + float(strong.std(ddof=0)))
    rel = diff / strong.mean() if strong.mean() > 0 else 0.0

    return {
        "runs": runs,
        "summary": summary,
        "flagship": FLAGSHIP,
        "baselines": list(BASELINE_METHODS),
        "gate": {
            "metric": "ndcg@10",
            "flagship_mean": float(flag.mean()),
            "flagship_std": float(flag.std(ddof=0)),
            "strong_baseline_mean": float(strong.mean()),
            "strong_baseline_std": float(strong.std(ddof=0)),
            "diff": diff,
            "relative_gain": rel,
            "threshold_rel": cfg.ndcg_gain_rel,
            "significant": bool(sig),
            "passed": bool(sig and rel >= cfg.ndcg_gain_rel),
        },
        "seeds": [int(s) for s in seeds],
    }


def summarize(bench: dict) -> str:
    """打印基准汇总表。`gate` 可缺省（单 seed `run` 场景无门禁）。"""
    lines = []
    g = bench.get("gate") or {}
    lines.append(f"seeds = {bench['seeds']}   metric = {g.get('metric', 'ndcg@10')}")
    lines.append(f"{'method':<12}{'nDCG@10':>18}{'nDCG@5':>18}{'MAP@10':>18}{'Recall@100':>18}")
    for m, metrics in bench["summary"].items():
        cells = []
        for k in ("ndcg@10", "ndcg@5", "map@10", "recall@100"):
            d = metrics.get(k)
            cells.append(f"{d['mean']:.4f}±{d['std']:.4f}" if d else "-")
        lines.append(f"{m:<12}" + "".join(f"{c:>18}" for c in cells))
    if "flagship_mean" in g:
        lines.append(
            f"gate: flagship {g['flagship_mean']:.4f} vs strong baseline "
            f"{g['strong_baseline_mean']:.4f} | Δ={g['diff']:+.4f} "
            f"({g['relative_gain']:+.2%}) | threshold {g['threshold_rel']:+.2%} | "
            f"significant={g['significant']} | PASS={g['passed']}"
        )
    return "\n".join(lines)
