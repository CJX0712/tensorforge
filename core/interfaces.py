"""分解器接口（Protocol）。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from core.types import DecompResult


@runtime_checkable
class Decomposer(Protocol):
    name: str

    def __call__(self, X: np.ndarray, ranks, seed: int | None = None, **kw) -> DecompResult: ...
