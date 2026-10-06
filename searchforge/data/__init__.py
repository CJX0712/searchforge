"""SearchForge · 数据层。

星 (晨星) · 2026-10-07
"""

from .generator import (
    SyntheticDataset,
    build_qrels,
    generate_corpus,
    generate_queries,
)

__all__ = ["generate_corpus", "generate_queries", "build_qrels", "SyntheticDataset"]
