"""SearchForge · 错误码（E100~E500）。

星 (晨星) · 2026-10-07
"""


class SearchForgeError(Exception):
    """所有 SearchForge 异常的基类。"""

    code = "E000"

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class E100ConfigError(SearchForgeError):
    code = "E100"


class E200DataError(SearchForgeError):
    code = "E200"


class E300RetrievalError(SearchForgeError):
    code = "E300"


class E400RerankError(SearchForgeError):
    code = "E400"


class E500EvalError(SearchForgeError):
    code = "E500"
