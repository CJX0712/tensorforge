"""pipeline 编排与确定性测试。"""

from __future__ import annotations

from core.config import Config
from core.seed import set_all
from pipeline.pipeline import TensorPipeline


def test_benchmark_structure():
    cfg = Config(sizes=(10, 10, 10), ranks=(3, 3, 3), noise_rel=0.1, n_seeds=3, seed=1)
    pipe = TensorPipeline(cfg)
    report = pipe.benchmark(out_path=None)
    assert len(report["methods"]) == 6
    assert len(report["per_seed"]) == 3
    names = {m["method"] for m in report["methods"]}
    assert {"naive", "matsvd", "hosvd", "tucker_hooi", "cp_als", "tensor_fuse"} <= names
    for m in report["methods"]:
        assert "mean_rel_err" in m


def test_determinism_two_runs_equal():
    cfg = Config(sizes=(10, 10, 10), ranks=(3, 3, 3), noise_rel=0.1, n_seeds=4, seed=2)
    pipe = TensorPipeline(cfg)
    r1 = pipe.benchmark(out_path=None)
    set_all(cfg.seed)
    r2 = pipe.benchmark(out_path=None)
    assert r1["per_seed"] == r2["per_seed"]


def test_failure_cases_derived():
    cfg = Config(sizes=(10, 10, 10), ranks=(3, 3, 3), noise_rel=0.1, seed=3)
    pipe = TensorPipeline(cfg)
    cases = pipe.failure_cases()
    assert len(cases) >= 3
    for c in cases:
        assert "id" in c and "title" in c and "cause" in c
