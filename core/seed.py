"""全局确定性入口。

唯一 seed 入口 ``set_all``：一次性设齐 numpy Generator / 传统 RandomState /
标准库 random，保证同 seed 两次运行逐位一致（确定性门禁的核心）。
"""

from __future__ import annotations

import random as _random

import numpy as np

_SEED: int | None = None


def set_all(seed: int) -> np.random.Generator:
    """设置全局确定性种子，返回 numpy Generator。

    所有随机源（numpy Generator、传统 np.random、标准库 random）均被同一
    seed 锁死，确保 benchmark 可 bit-for-bit 复现。
    """
    global _SEED
    _SEED = int(seed)
    rng = np.random.default_rng(_SEED)
    np.random.seed(_SEED)
    _random.seed(_SEED)
    return rng


def get_seed() -> int:
    return _SEED if _SEED is not None else 0


def rng_for(seed: int | None = None) -> np.random.Generator:
    """取得一个独立的 numpy Generator（不污染全局状态）。"""
    if seed is None:
        seed = get_seed()
    return np.random.default_rng(seed)
