"""评测指标。统一口径：误差越小越好。"""

from __future__ import annotations

import numpy as np


def rel_err(reconstructed: np.ndarray, true: np.ndarray) -> float:
    """相对 Frobenius 误差 ``‖R−T‖/‖T‖``。"""
    diff = np.asarray(reconstructed, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    denom = np.linalg.norm(true)
    if denom == 0:
        raise ValueError("[E200] 真值张量范数为 0，无法计算相对误差")
    return float(np.linalg.norm(diff) / denom)


def rmse(reconstructed: np.ndarray, true: np.ndarray) -> float:
    """均方根误差。"""
    diff = np.asarray(reconstructed, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean(diff**2)))


def reduction(numerator: float, denominator: float) -> float:
    """相对下降比例 ``1 − num/den``（正数表示下降）。"""
    if denominator == 0:
        return float("nan")
    return float(1.0 - numerator / denominator)
