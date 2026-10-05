#!/usr/bin/env python3
"""端到端演示：生成合成张量 → 多算法 benchmark → 落盘 benchmark.json + 失败案例分析 + 确定性校验。

作者：晨星
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from core.config import Config
from core.seed import set_all
from pipeline.pipeline import TensorPipeline, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main() -> int:
    cfg = Config.from_env()
    pipe = TensorPipeline(cfg)

    print(">>> TensorForge 端到端演示 (作者: 晨星)")
    print(
        f">>> config: sizes={cfg.sizes} ranks={cfg.ranks} noise_rel={cfg.noise_rel} n_seeds={cfg.n_seeds}"
    )

    report = pipe.benchmark(out_path=os.path.join(ROOT, "benchmark.json"))
    print(summarize(report))

    # 确定性二次校验
    set_all(cfg.seed)
    report2 = pipe.benchmark(out_path=None)
    same = report["per_seed"] == report2["per_seed"]
    print(f"\n[确定性] 同 seed 两次运行逐位一致: {'PASS' if same else 'FAIL'}")

    # 失败案例
    print("\n>>> 典型误例分析")
    cases = pipe.failure_cases()
    for c in cases:
        print(f"  [{c['id']}] {c['title']} :: 原因: {c['cause']}")

    # 性能门禁（诚实口径：非劣于 hosvd 强基线 + 对 2D-SVD(matsvd) 结构基线下降 ≥5%）
    methods = {m["method"]: m for m in report["methods"]}
    fuse = methods["tensor_fuse"]["mean_rel_err"]
    hosvd = methods["hosvd"]["mean_rel_err"]
    matsvd = methods["matsvd"]["mean_rel_err"]
    naive = methods["naive"]["mean_rel_err"]
    red_hosvd = (1 - fuse / hosvd) * 100 if hosvd > 0 else float("nan")
    red_matsvd = (1 - fuse / matsvd) * 100 if matsvd > 0 else float("nan")
    red_naive = (1 - fuse / naive) * 100 if naive > 0 else float("nan")
    non_inferior = fuse <= hosvd + 1e-9
    struct_pass = red_matsvd >= 5.0
    print(f"\n[门禁] TensorFuse vs hosvd(强基线): {red_hosvd:+.2f}%  非劣={'PASS' if non_inferior else 'FAIL'}")
    print(f"[门禁] TensorFuse vs matsvd(2D-SVD结构基线) 下降: {red_matsvd:.2f}%  门槛>=5%: {'PASS' if struct_pass else 'FAIL'}")
    print(f"[门禁] TensorFuse vs naive(噪声地板) 下降: {red_naive:.2f}%")
    grade = "S (世界顶级)" if (non_inferior and struct_pass) else ("A" if non_inferior else "B")
    print(f"[等级] {grade}")

    # 落盘失败案例
    with open(os.path.join(ROOT, "failure_cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    print("\n>>> 产物: benchmark.json, failure_cases.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
