"""确定性门禁：同 seed 逐位一致。"""

from __future__ import annotations

import numpy as np

from core.seed import set_all
from data.synthetic import make_tucker_tensor
from tensor.decompositions import tucker_hooi


def test_synthetic_generation_bit_reproducible():
    a = make_tucker_tensor((10, 10, 10), (3, 3, 3), 0.1, seed=123)[0]
    b = make_tucker_tensor((10, 10, 10), (3, 3, 3), 0.1, seed=123)[0]
    assert np.array_equal(a, b)


def test_method_bit_reproducible():
    X = make_tucker_tensor((10, 10, 10), (3, 3, 3), 0.1, seed=55)[0]
    set_all(55)
    r1 = tucker_hooi(X, (3, 3, 3)).reconstructed
    set_all(55)
    r2 = tucker_hooi(X, (3, 3, 3)).reconstructed
    assert np.array_equal(r1, r2)
