"""张量分解算法集。

含：
- ``naive``       : 不压缩（重构=观测），代表噪声地板（下界参考）
- ``matsvd``      : 仅对模式 0 做 2D 截断 SVD（忽略张量多向结构，朴素基线）
- ``hosvd``       : 高阶 SVD 单遍 Tucker（标准非迭代基线）
- ``tucker_hooi`` : Higher-Order Orthogonal Iteration（Tucker 经典 SOTA）
- ``cp_als``      : CP/PARAFAC 交替最小二乘（CP 经典 SOTA）

全部纯 numpy，零外部依赖，确定性可复现。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from core.errors import RankError
from core.types import DecompResult
from preprocess.unfold import (
    fold,
    khatri_rao_chain,
    kruskal_to_tensor,
    ttm,
    unfold,
)


def _check_ranks(X: np.ndarray, ranks: Sequence[int]) -> tuple[int, ...]:
    ranks = tuple(int(r) for r in ranks)
    if len(ranks) != X.ndim:
        raise RankError(f"ranks 长度 {len(ranks)} 与维数 {X.ndim} 不一致")
    for s, r in zip(X.shape, ranks):
        if not (1 <= r <= s):
            raise RankError(f"秩 {r} 超出维度 {s}")
    return ranks


def naive(
    X: np.ndarray, ranks: Sequence[int] = (), seed: Optional[int] = None, **kw
) -> DecompResult:
    """不压缩：重构即观测张量（噪声地板参考）。"""
    return DecompResult("naive", X.astype(np.float64, copy=True), n_iter=0)


def matsvd(X: np.ndarray, ranks: Sequence[int], seed: Optional[int] = None, **kw) -> DecompResult:
    """朴素基线：仅对模式 0 做 2D 截断 SVD，忽略其余维度结构。"""
    ranks = _check_ranks(X, ranks)
    r = ranks[0]
    U0 = unfold(X, 0)
    U, S, Vt = np.linalg.svd(U0, full_matrices=False)
    rec = (U[:, :r] * S[:r]) @ Vt[:r, :]
    Xr = fold(rec, 0, X.shape)
    return DecompResult("matsvd", Xr, factors=[U[:, :r]], n_iter=1)


def hosvd(X: np.ndarray, ranks: Sequence[int], seed: Optional[int] = None, **kw) -> DecompResult:
    """高阶 SVD（单遍 Tucker），标准非迭代基线。"""
    ranks = _check_ranks(X, ranks)
    N = X.ndim
    Us = []
    for n in range(N):
        Un = unfold(X, n)
        U, _, _ = np.linalg.svd(Un, full_matrices=False)
        Us.append(U[:, : ranks[n]])
    G = X
    for k in range(N):
        G = ttm(G, Us[k].T, k)
    Xr = G
    for k in range(N):
        Xr = ttm(Xr, Us[k], k)
    return DecompResult("hosvd", Xr, core=G, factors=Us, n_iter=0)


def tucker_hooi(
    X: np.ndarray,
    ranks: Sequence[int],
    seed: Optional[int] = None,
    n_iter: int = 30,
    tol: float = 1e-8,
    **kw,
) -> DecompResult:
    """Tucker-HOOI：交替正交迭代，Tucker 分解的经典 SOTA。

    每轮对模式 n 投影 ``Y = X ×_k U_kᵀ (k≠n)``，取 ``Y₍ₙ₎`` 前 Rₙ 左奇异向量更新 Uₙ，
    直至重构相对残差变化低于 ``tol`` 或达 ``n_iter``。
    """
    ranks = _check_ranks(X, ranks)
    N = X.ndim
    Us = []
    for n in range(N):
        Un = unfold(X, n)
        U, _, _ = np.linalg.svd(Un, full_matrices=False)
        Us.append(U[:, : ranks[n]])
    prev = None
    it = 0
    for it in range(1, n_iter + 1):
        for n in range(N):
            Z = X
            for k in range(N):
                if k != n:
                    Z = ttm(Z, Us[k].T, k)
            Zn = unfold(Z, n)
            U, _, _ = np.linalg.svd(Zn, full_matrices=False)
            Us[n] = U[:, : ranks[n]]
        G = X
        for k in range(N):
            G = ttm(G, Us[k].T, k)
        Xr = G
        for k in range(N):
            Xr = ttm(Xr, Us[k], k)
        err = float(np.linalg.norm(Xr - X) / np.linalg.norm(X))
        if prev is not None and abs(prev - err) < tol:
            break
        prev = err
    G = X
    for k in range(N):
        G = ttm(G, Us[k].T, k)
    Xr = G
    for k in range(N):
        Xr = ttm(Xr, Us[k], k)
    return DecompResult("tucker_hooi", Xr, core=G, factors=Us, n_iter=it)


def cp_als(
    X: np.ndarray,
    ranks: Sequence[int],
    seed: Optional[int] = None,
    n_iter: int = 200,
    tol: float = 1e-8,
    **kw,
) -> DecompResult:
    """CP/PARAFAC 交替最小二乘（CP 经典 SOTA）。

    CP 秩取 ``ranks[0]``（各模式统一秩）。逐模式用其余因子的 Khatri-Rao 链做最小二乘更新。
    """
    ranks = _check_ranks(X, ranks)
    N = X.ndim
    R = ranks[0]
    rng = np.random.default_rng(0 if seed is None else int(seed))
    factors = [rng.standard_normal((X.shape[n], R)) for n in range(N)]
    prev = None
    it = 0
    for it in range(1, n_iter + 1):
        for n in range(N):
            others = [factors[k] for k in range(N) if k != n]
            KR = khatri_rao_chain(others)
            Xn = unfold(X, n)
            A = np.linalg.lstsq(KR, Xn.T, rcond=None)[0].T
            factors[n] = A
        Xr = kruskal_to_tensor(factors)
        err = float(np.linalg.norm(Xr - X) / np.linalg.norm(X))
        if prev is not None and abs(prev - err) < tol:
            break
        prev = err
    Xr = kruskal_to_tensor(factors)
    return DecompResult("cp_als", Xr, factors=factors, n_iter=it)


# 注册表：名称 -> 可调用
DECOMPOSERS = {
    "naive": naive,
    "matsvd": matsvd,
    "hosvd": hosvd,
    "tucker_hooi": tucker_hooi,
    "cp_als": cp_als,
}


def available_methods() -> list[str]:
    return list(DECOMPOSERS.keys())


def run_method(name: str, X: np.ndarray, ranks, seed: Optional[int] = None, **kw) -> DecompResult:
    if name not in DECOMPOSERS:
        raise RankError(f"未知分解方法: {name}")
    return DECOMPOSERS[name](X, ranks, seed=seed, **kw)
