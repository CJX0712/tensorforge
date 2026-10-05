"""分解算法正确性测试。"""

from __future__ import annotations

import numpy as np

from core.errors import RankError
from core.seed import set_all
from data.synthetic import make_tucker_tensor
from preprocess.unfold import kruskal_to_tensor
from tensor.decompositions import (
    cp_als,
    hosvd,
    matsvd,
    naive,
    run_method,
    tucker_hooi,
)


def _tucker(sizes=(10, 10, 10), ranks=(3, 3, 3), seed=0):
    set_all(seed)
    X_obs, X_true = make_tucker_tensor(sizes, ranks, noise_rel=0.0, seed=seed)
    return X_obs, X_true


def test_exact_tucker_recovery():
    X, X_true = _tucker()
    r = hosvd(X, (3, 3, 3))
    r.score_against(X_true)
    assert r.rel_err < 1e-9, f"hosvd 应精确恢复无噪 Tucker, got {r.rel_err}"


def test_hooi_not_worse_than_hosvd():
    set_all(3)
    X_obs, X_true = make_tucker_tensor((12, 12, 12), (3, 3, 3), noise_rel=0.1, seed=3)
    h = hosvd(X_obs, (3, 3, 3)).score_against(X_true)
    ho = tucker_hooi(X_obs, (3, 3, 3), n_iter=30).score_against(X_true)
    assert ho.rel_err <= h.rel_err + 1e-6


def test_exact_cp_recovery():
    rng = np.random.default_rng(11)
    R = 3
    factors = [
        rng.standard_normal((8, R)),
        rng.standard_normal((9, R)),
        rng.standard_normal((10, R)),
    ]
    X = kruskal_to_tensor(factors)
    r = cp_als(X, (R, R, R), seed=11, n_iter=500, tol=1e-10)
    r.score_against(X)
    assert r.rel_err < 1e-6, f"cp_als 应精确恢复无噪 CP, got {r.rel_err}"


def test_matsvd_weaker_than_hooi_on_tucker():
    set_all(5)
    X_obs, X_true = make_tucker_tensor((16, 16, 16), (3, 3, 3), noise_rel=0.12, seed=5)
    m = matsvd(X_obs, (3, 3, 3)).score_against(X_true)
    ho = tucker_hooi(X_obs, (3, 3, 3)).score_against(X_true)
    # 2D 截断 SVD 忽略张量联合结构，应明显差于 HOOI
    assert m.rel_err > ho.rel_err * 1.2


def test_naive_is_noise_floor():
    set_all(7)
    X_obs, X_true = make_tucker_tensor((12, 12, 12), (3, 3, 3), noise_rel=0.1, seed=7)
    n = naive(X_obs).score_against(X_true)
    assert abs(n.rel_err - 0.1) < 1e-6


def test_bad_ranks_raise():
    X = np.zeros((5, 5, 5))
    try:
        run_method("hosvd", X, (3, 3, 9))
        assert False, "应抛 RankError"
    except RankError:
        pass
