"""SearchForge · 学习重排（LambdaMART / LightGBM，顶级开源；缺失则 sklearn 兜底）。

星 (晨星) · 2026-10-07

LambdaMART（Burges et al. 2006/2007）是工业界主导的学习排序算法（Bing 等）。
这里以 (query, candidate) 的特征为输入、分级相关性为标签、按 query 分组训练，
对初排候选列表重排序。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E400RerankError
from ..core.interfaces import Reranker
from ..core.types import Corpus, Qrels
from .features import FEATURE_NAMES


def _as_matrix(rows: list[dict], names: list[str] | None = None) -> np.ndarray:
    names = names or FEATURE_NAMES
    return np.asarray([[r[name] for name in names] for r in rows], dtype=np.float64)


class LightGBMReranker:
    """LightGBM LambdaMART 重排器（确定性：单线程、关闭 bagging）。"""

    name = "rankfuse_lambdamart"

    def __init__(
        self,
        random_state: int = 42,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        n_estimators: int = 200,
        feature_names: list[str] | None = None,
    ) -> None:
        self.random_state = random_state
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.feature_names = feature_names or list(FEATURE_NAMES)
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import lightgbm  # noqa: F401

            return True
        except Exception:
            return False

    def fit(
        self, corpus: Corpus, qrels: Qrels, train_qids: list[int], feat_by_qid: dict
    ) -> "LightGBMReranker":
        if not self.available():
            raise E400RerankError("lightgbm 不可用")
        import lightgbm as lgb

        X, y, groups = [], [], []
        for qid in train_qids:
            rows = feat_by_qid.get(qid, [])
            if not rows:
                continue
            for r, d in rows:
                X.append([r[name] for name in self.feature_names])
                y.append(qrels.grade(qid, d))
            groups.append(len(rows))
        if not X:
            raise E400RerankError("训练集为空")
        if max(y) <= 0:
            raise E400RerankError("训练集无正样本（全 0 标签）")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        self._model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[10],
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,
            bagging_fraction=1.0,
            bagging_freq=0,
            feature_fraction=1.0,
            min_child_samples=1,
            verbose=-1,
        )
        self._model.fit(X, y, group=groups)
        return self

    def rerank(self, qid: int, candidates: dict) -> list[int]:
        if self._model is None:
            raise E400RerankError("未 fit")
        doc_ids = list(candidates.keys())
        if not doc_ids:
            return []
        X = _as_matrix([candidates[d] for d in doc_ids], self.feature_names)
        scores = np.asarray(self._model.predict(X), dtype=np.float64)
        order = np.argsort(-scores)
        return [doc_ids[i] for i in order]


class SklearnRanker:
    """离线兜底：点对点 LogisticRegression（label = grade>0）作为重排分。"""

    name = "rankfuse_sklearn"

    def __init__(self, random_state: int = 42, feature_names: list[str] | None = None) -> None:
        self.random_state = random_state
        self.feature_names = feature_names or list(FEATURE_NAMES)
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import sklearn  # noqa: F401

            return True
        except Exception:
            return False

    def fit(
        self, corpus: Corpus, qrels: Qrels, train_qids: list[int], feat_by_qid: dict
    ) -> "SklearnRanker":
        from sklearn.linear_model import LogisticRegression

        X, y = [], []
        for qid in train_qids:
            for r, d in feat_by_qid.get(qid, []):
                X.append([r[name] for name in self.feature_names])
                y.append(1 if qrels.grade(qid, d) > 0 else 0)
        if not X or max(y) <= 0:
            raise E400RerankError("sklearn 兜底训练集无正样本")
        self._model = LogisticRegression(random_state=self.random_state, max_iter=200)
        self._model.fit(np.asarray(X), np.asarray(y))
        return self

    def rerank(self, qid: int, candidates: dict) -> list[int]:
        if self._model is None:
            raise E400RerankError("未 fit")
        doc_ids = list(candidates.keys())
        if not doc_ids:
            return []
        X = _as_matrix([candidates[d] for d in doc_ids], self.feature_names)
        scores = self._model.predict_proba(X)[:, 1]
        order = np.argsort(-scores)
        return [doc_ids[i] for i in order]


def build_reranker(random_state: int = 42, **kwargs) -> Reranker:
    """工厂：优先 LightGBM LambdaMART，否则 sklearn 兜底。"""
    if LightGBMReranker.available():
        return LightGBMReranker(random_state=random_state, **kwargs)
    return SklearnRanker(random_state=random_state)
