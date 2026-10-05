"""unfold / fold / ttm / kruskal 数学不变量测试。"""

from __future__ import annotations

import numpy as np

from preprocess.unfold import (
    fold,
    khatri_rao,
    kruskal_to_tensor,
    ttm,
    unfold,
)


def test_unfold_fold_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 6, 7))
    for n in range(3):
        U = unfold(X, n)
        X2 = fold(U, n, X.shape)
        assert np.allclose(X, X2)


def test_ttm_orthogonal_identity():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5, 6, 7))
    # 用正交矩阵乘模式 0 再乘其转置应近似还原
    Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    Y = ttm(ttm(X, Q, 0), Q.T, 0)
    assert np.allclose(X, Y, atol=1e-10)


def test_ttm_shape():
    X = np.zeros((4, 5, 6))
    Y = ttm(X, np.zeros((3, 4)), 0)
    assert Y.shape == (3, 5, 6)


def test_khatri_rao_shape():
    A = np.ones((4, 3))
    B = np.ones((5, 3))
    K = khatri_rao(A, B)
    assert K.shape == (20, 3)


def test_kruskal_reconstruct_identity_factors():
    # 全 1 因子 → 重构为全 R 的张量
    factors = [np.ones((3, 2)), np.ones((4, 2)), np.ones((5, 2))]
    T = kruskal_to_tensor(factors)
    assert T.shape == (3, 4, 5)
    assert np.allclose(T, 2.0)
