"""核心数据类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DecompResult:
    """单次分解结果。

    ``reconstructed`` 为重构张量；``rel_err`` / ``rmse`` 仅在与真值比对后填充。
    """

    name: str
    reconstructed: np.ndarray
    core: Optional[np.ndarray] = None
    factors: List[np.ndarray] = field(default_factory=list)
    rel_err: float = float("nan")
    rmse: float = float("nan")
    n_iter: int = 0
    extra: dict = field(default_factory=dict)

    def score_against(self, true: np.ndarray) -> "DecompResult":
        diff = self.reconstructed - true
        denom = np.linalg.norm(true)
        self.rel_err = float(np.linalg.norm(diff) / denom) if denom > 0 else float("nan")
        self.rmse = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
        return self


@dataclass
class BenchRow:
    method: str
    rel_err: float
    rmse: float
    n_iter: int
    skipped: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "rel_err": self.rel_err,
            "rmse": self.rmse,
            "n_iter": self.n_iter,
            "skipped": self.skipped,
            "note": self.note,
        }


@dataclass
class Aggregate:
    method: str
    mean_rel_err: float
    std_rel_err: float
    mean_rmse: float
    std_rmse: float
    mean_n_iter: float
    n_seeds: int

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "mean_rel_err": self.mean_rel_err,
            "std_rel_err": self.std_rel_err,
            "mean_rmse": self.mean_rmse,
            "std_rmse": self.std_rmse,
            "mean_n_iter": self.mean_n_iter,
            "n_seeds": self.n_seeds,
        }
