"""旗舰 ``TensorFuse``：Tucker-HOOI 与 CP-ALS 自适应择优融合。

选择准则：在**观测张量**上比较两路重构残差（``‖X_rec − X_obs‖``），取残差更小者。
真值仅用于 benchmark 评测打分，不参与选择，杜绝信息泄漏。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from core.types import DecompResult
from tensor.decompositions import cp_als, tucker_hooi


def tensor_fuse(
    X: np.ndarray,
    ranks: Sequence[int],
    seed: Optional[int] = None,
    n_iter: int = 30,
    cp_iters: int = 200,
    tol: float = 1e-8,
    **kw,
) -> DecompResult:
    """Tucker-HOOI 与 CP-ALS 自适应择优。

    返回被选中方法的 ``DecompResult``（name 重命名为 ``tensor_fuse``），
    并在 ``extra`` 中记录两路残差与最终选择，便于可解释审计。
    """
    hooi = tucker_hooi(X, ranks, seed=seed, n_iter=n_iter, tol=tol, **kw)
    cp = cp_als(X, ranks, seed=seed, n_iter=cp_iters, tol=tol, **kw)

    rh = float(np.linalg.norm(hooi.reconstructed - X) / np.linalg.norm(X))
    rc = float(np.linalg.norm(cp.reconstructed - X) / np.linalg.norm(X))

    if rc <= rh:
        chosen = cp
        choice = "cp_als"
    else:
        chosen = hooi
        choice = "tucker_hooi"

    chosen.name = "tensor_fuse"
    chosen.extra = {
        "selection": choice,
        "hooi_resid": rh,
        "cp_resid": rc,
        "hooi_n_iter": hooi.n_iter,
        "cp_n_iter": cp.n_iter,
    }
    return chosen


def tensor_fuse_hooi_only(
    X: np.ndarray, ranks: Sequence[int], seed: Optional[int] = None, **kw
) -> DecompResult:
    """消融变体：固定只用 Tucker-HOOI（关闭 CP 备选）。"""
    hooi = tucker_hooi(X, ranks, seed=seed, **kw)
    hooi.name = "tensor_fuse_hooi_only"
    hooi.extra = {"selection": "tucker_hooi", "note": "ablation: CP alternative disabled"}
    return hooi
