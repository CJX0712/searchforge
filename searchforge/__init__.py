"""SearchForge · 世界级神经/混合检索与学习重排系统。

星 (晨星) · 2026-10-07

复用顶级开源：FAISS(ANN) · rank_bm25(BM25) · LightGBM(LambdaMART) · Optuna(HPO) · scikit-learn。
旗舰 RankFuse = 双路召回(BM25 + LSI/FAISS) → RRF 融合 → LambdaMART 学习重排。
"""

from .core import Config, set_all
from .pipeline import SearchPipeline, benchmark

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = ["set_all", "Config", "SearchPipeline", "benchmark", "__version__"]
