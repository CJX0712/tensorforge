"""seed 全局确定性入口测试。"""

from __future__ import annotations

import numpy as np

from core.seed import get_seed, rng_for, set_all


def test_set_all_sets_global():
    rng = set_all(42)
    assert get_seed() == 42
    a = rng.standard_normal(5)
    set_all(42)
    b = rng_for(42).standard_normal(5)
    assert np.array_equal(a, b)
