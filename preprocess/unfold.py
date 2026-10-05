"""张量展开/折叠与张量-矩阵乘法（N 阶通用）。

所有运算纯 numpy 实现，零外部依赖，确定性可复现。是张量分解的数学基座。
"""

from __future__ import annotations

from functools import reduce

import numpy as np


def unfold(X: np.ndarray, n: int) -> np.ndarray:
    """沿模式 n 将张量展开为矩阵。

    返回形状 ``(X.shape[n], prod(其余维度))``。
    """
    N = X.ndim
    if not (0 <= n < N):
        raise IndexError(f"模式索引 {n} 超出维度数 {N}")
    order = [n] + [i for i in range(N) if i != n]
    Xr = np.transpose(X, order)
    return Xr.reshape(X.shape[n], -1)


def fold(Un: np.ndarray, n: int, shape: tuple) -> np.ndarray:
    """``unfold`` 的逆操作，按 ``shape`` 还原张量。"""
    N = len(shape)
    others = [shape[i] for i in range(N) if i != n]
    Xr = Un.reshape(shape[n], *others)
    order = [n] + [i for i in range(N) if i != n]
    inv = [0] * N
    for i in range(N):
        inv[order[i]] = i
    return np.transpose(Xr, inv)


def ttm(X: np.ndarray, M: np.ndarray, n: int) -> np.ndarray:
    """张量-矩阵乘法：``X ×_n M``，将模式 n 维 ``X.shape[n]`` 映射为 ``M.shape[0]``。"""
    Un = unfold(X, n)
    Yn = M @ Un
    new_shape = tuple(M.shape[0] if i == n else X.shape[i] for i in range(X.ndim))
    return fold(Yn, n, new_shape)


def khatri_rao(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """列向 Kronecker 积（Khatri-Rao 积）。"""
    return np.einsum("ik,jk->ijk", A, B).reshape(A.shape[0] * B.shape[0], A.shape[1])


def khatri_rao_chain(mats: list[np.ndarray]) -> np.ndarray:
    """多矩阵 Khatri-Rao 链积（按列表顺序）。"""
    return reduce(khatri_rao, mats)


def kruskal_to_tensor(factors: list[np.ndarray]) -> np.ndarray:
    """由 CP 因子矩阵重建张量：``Σ_r a_r⁽¹⁾ ∘ … ∘ a_r⁽ᴺ⁾``。

    采用动态 einsum 规格 ``i0r,i1r,...,i{N-1}r -> i0,i1,...,i{N-1}r``，
    避免逐维累乘时下标维度不匹配。
    """
    N = len(factors)
    if N == 1:
        return factors[0].astype(np.float64, copy=True)
    letters = [chr(ord("a") + n) for n in range(N)]
    subs = [l + "r" for l in letters]
    spec = ",".join(subs) + "->" + "".join(letters) + "r"
    T = np.einsum(spec, *factors)
    return np.sum(T, axis=-1)


def tucker_to_tensor(core: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
    """由 Tucker 核心与因子矩阵重建张量。"""
    X = core
    for k in range(len(factors)):
        X = ttm(X, factors[k], k)
    return X
