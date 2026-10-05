"""CLI 冒烟测试（进程内调用 main）。"""

from __future__ import annotations

import os
import tempfile

from cli import main


def test_cli_selftest():
    rc = main(["selftest", "--sizes", "8,8,8", "--ranks", "2,2,2", "--seed", "1"])
    assert rc == 0


def test_cli_run():
    rc = main(["run", "--sizes", "8,8,8", "--ranks", "2,2,2", "--noise-rel", "0.1", "--seed", "2"])
    assert rc == 0


def test_cli_failure():
    rc = main(["failure", "--sizes", "8,8,8", "--ranks", "2,2,2"])
    assert rc == 0


def test_cli_benchmark_writes():
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "bench.json")
    rc = main(["benchmark", "--sizes", "8,8,8", "--ranks", "2,2,2", "--n-seeds", "2", "--out", out])
    assert rc == 0
    assert os.path.exists(out)
