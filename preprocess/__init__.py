"""preprocess：张量预处理层（展开/折叠/张量乘法）。"""

from __future__ import annotations

from preprocess.unfold import (
    fold,
    khatri_rao,
    khatri_rao_chain,
    kruskal_to_tensor,
    ttm,
    tucker_to_tensor,
    unfold,
)

__all__ = [
    "fold",
    "khatri_rao",
    "khatri_rao_chain",
    "kruskal_to_tensor",
    "ttm",
    "tucker_to_tensor",
    "unfold",
]
