"""SearchForge · 全局确定性入口。

星 (晨星) · 2026-10-07

唯一 seed 入口：一次设齐 python / numpy / 环境变量。FAISS(LSH/Flat) 与 LightGBM
均为确定性算法，分别通过构造参数与 ``random_state`` 控制，无需全局随机状态。
"""

from __future__ import annotations

import os
import random
from typing import Optional

_SEED: Optional[int] = None


def set_all(seed: int = 42) -> int:
    """设置全局确定性种子，返回实际生效的种子。"""
    global _SEED
    _SEED = int(seed)
    random.seed(_SEED)
    try:
        import numpy as np

        np.random.seed(_SEED)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(_SEED)
    return _SEED


def get_seed() -> int:
    return _SEED if _SEED is not None else 42
