"""旗舰 TensorFuse 测试。"""

from __future__ import annotations

from core.seed import set_all
from data.synthetic import make_tucker_tensor
from tensor.fuse import tensor_fuse, tensor_fuse_hooi_only


def test_tensor_fuse_basic():
    set_all(2)
    X_obs, X_true = make_tucker_tensor((12, 12, 12), (3, 3, 3), noise_rel=0.1, seed=2)
    r = tensor_fuse(X_obs, (3, 3, 3), seed=2)
    assert r.name == "tensor_fuse"
    assert r.reconstructed.shape == X_obs.shape
    assert "selection" in r.extra
    assert r.extra["selection"] in ("tucker_hooi", "cp_als")
    r.score_against(X_true)
    assert r.rel_err < 0.1


def test_tensor_fuse_hooi_only():
    set_all(4)
    X_obs, X_true = make_tucker_tensor((10, 10, 10), (3, 3, 3), noise_rel=0.1, seed=4)
    r = tensor_fuse_hooi_only(X_obs, (3, 3, 3), seed=4)
    assert r.name == "tensor_fuse_hooi_only"
    assert r.reconstructed.shape == X_obs.shape


def test_tensor_fuse_noninferior_to_baselines():
    set_all(9)
    X_obs, X_true = make_tucker_tensor((14, 14, 14), (3, 3, 3), noise_rel=0.12, seed=9)
    from tensor.decompositions import cp_als, hosvd, tucker_hooi

    base = {
        "hosvd": hosvd(X_obs, (3, 3, 3)).score_against(X_true).rel_err,
        "tucker_hooi": tucker_hooi(X_obs, (3, 3, 3)).score_against(X_true).rel_err,
        "cp_als": cp_als(X_obs, (3, 3, 3)).score_against(X_true).rel_err,
    }
    fuse = tensor_fuse(X_obs, (3, 3, 3), seed=9).score_against(X_true).rel_err
    # 自适应择优应不劣于任一分量的重构误差
    assert fuse <= min(base.values()) + 1e-9
