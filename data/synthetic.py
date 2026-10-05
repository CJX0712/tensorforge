"""合成数据生成（确定性、可复现）。

主生成器 ``make_tucker_tensor``：低 Tucker 秩张量 + 可控相对噪声。
噪声水平按 ``noise_rel = ‖noise‖/‖X_true‖`` 归一化，使 naive 基线重构误差恰为 noise_rel，
便于跨 seed 公平比较。另提供非张量结构张量（失败案例分析用）。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from preprocess.unfold import ttm


def _orthogonal_factors(
    sizes: Sequence[int], ranks: Sequence[int], rng: np.random.Generator
) -> list[np.ndarray]:
    Us = []
    for n in range(len(sizes)):
        M = rng.standard_normal((sizes[n], ranks[n]))
        Q, _ = np.linalg.qr(M)
        Us.append(Q)
    return Us


def make_tucker_tensor(
    sizes: Sequence[int],
    ranks: Sequence[int],
    noise_rel: float = 0.12,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    core_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成低 Tucker 秩张量并叠加可控相对噪声。

    返回 ``(X_obs, X_true)``：
    - ``X_true``：单位 Frobenius 范数的低秩真值张量
    - ``X_obs``：``X_true + noise``，其中 ``‖noise‖/‖X_true‖ = noise_rel``
    """
    if rng is None:
        rng = np.random.default_rng(0 if seed is None else int(seed))
    sizes = tuple(int(s) for s in sizes)
    ranks = tuple(int(r) for r in ranks)
    N = len(sizes)
    Us = _orthogonal_factors(sizes, ranks, rng)
    core = rng.standard_normal(ranks) * core_scale
    Xtrue = core.astype(np.float64)
    for k in range(N):
        Xtrue = ttm(Xtrue, Us[k], k)
    Xtrue = Xtrue / np.linalg.norm(Xtrue)

    noise = rng.standard_normal(sizes)
    noise = noise / np.linalg.norm(noise) * noise_rel * np.linalg.norm(Xtrue)
    Xobs = Xtrue + noise
    return Xobs, Xtrue


def make_random_tensor(
    sizes: Sequence[int],
    noise_rel: float = 0.12,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """非低秩结构张量（高斯白噪声张量）。用于失败案例分析（Tucker 假设不成立）。"""
    if rng is None:
        rng = np.random.default_rng(0 if seed is None else int(seed))
    sizes = tuple(int(s) for s in sizes)
    Xtrue = rng.standard_normal(sizes)
    Xtrue = Xtrue / np.linalg.norm(Xtrue)
    noise = rng.standard_normal(sizes)
    noise = noise / np.linalg.norm(noise) * noise_rel * np.linalg.norm(Xtrue)
    return Xtrue + noise, Xtrue
