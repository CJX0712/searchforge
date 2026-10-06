"""SearchForge · 核心层。

星 (晨星) · 2026-10-07
"""

from .config import Config
from .errors import (
    E100ConfigError,
    E200DataError,
    E300RetrievalError,
    E400RerankError,
    E500EvalError,
    SearchForgeError,
)
from .interfaces import Fusion, Reranker, Retriever
from .seed import get_seed, set_all
from .types import (
    Candidate,
    Corpus,
    Document,
    Qrels,
    Query,
    RetrievalResult,
)

__all__ = [
    "Document",
    "Query",
    "Corpus",
    "Qrels",
    "Candidate",
    "RetrievalResult",
    "SearchForgeError",
    "E100ConfigError",
    "E200DataError",
    "E300RetrievalError",
    "E400RerankError",
    "E500EvalError",
    "Config",
    "set_all",
    "get_seed",
    "Retriever",
    "Reranker",
    "Fusion",
]
