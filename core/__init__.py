"""TensorForge 核心包。"""

from __future__ import annotations

from core.config import Config
from core.errors import (
    ConfigError,
    ConvergenceError,
    DataLeakError,
    NotFittedError,
    RankError,
    ShapeMismatchError,
    TensorForgeError,
)
from core.seed import get_seed, rng_for, set_all
from core.types import Aggregate, BenchRow, DecompResult

__all__ = [
    "Aggregate",
    "BenchRow",
    "Config",
    "ConfigError",
    "ConvergenceError",
    "DataLeakError",
    "DecompResult",
    "NotFittedError",
    "RankError",
    "ShapeMismatchError",
    "TensorForgeError",
    "get_seed",
    "rng_for",
    "set_all",
]

__author__ = "晨星"
