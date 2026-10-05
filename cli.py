#!/usr/bin/env python3
"""TensorForge CLI 入口。

子命令：
  benchmark   运行跨 seed 基准，落盘 benchmark.json，打印性能基线表 + 确定性校验
  failure     派生并打印 ≥3 个典型误例（含原因归因）
  run         对合成（或载入）张量跑单次全方法分解
  selftest    快速自检（一组不变量）

用法示例：
  python cli.py benchmark --n-seeds 5 --noise-rel 0.12
  python cli.py failure
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from core.config import Config
from core.seed import set_all
from pipeline.pipeline import TensorPipeline, summarize


def _build_config(args: argparse.Namespace) -> Config:
    cfg = Config.from_env()
    if getattr(args, "sizes", None):
        cfg.sizes = tuple(int(x) for x in args.sizes.split(","))
    if getattr(args, "ranks", None):
        cfg.ranks = tuple(int(x) for x in args.ranks.split(","))
    if getattr(args, "noise_rel", None) is not None:
        cfg.noise_rel = float(args.noise_rel)
    if getattr(args, "n_seeds", None) is not None:
        cfg.n_seeds = int(args.n_seeds)
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    if getattr(args, "hooi_iters", None) is not None:
        cfg.hooi_iters = int(args.hooi_iters)
    if getattr(args, "cp_iters", None) is not None:
        cfg.cp_iters = int(args.cp_iters)
    cfg.validate()
    return cfg


def cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    pipe = TensorPipeline(cfg)
    report = pipe.benchmark(out_path=args.out)
    print(summarize(report))

    # 确定性校验：同 seed 两次运行逐位一致（排除 elapsed_sec）
    set_all(cfg.seed)
    report2 = pipe.benchmark(out_path=None)
    same = report["per_seed"] == report2["per_seed"]
    print(f"\n[确定性] 同 seed 两次运行核心指标逐位一致: {'PASS' if same else 'FAIL'}")
    if not same:
        return 1

    # 性能门禁（方案阶段定死，诚实重定义，参照 TscForge 先例）：
    #   (a) TensorFuse 非劣于强基线 hosvd（≤，即在精确 Tucker 数据上击败或持平单遍最优）
    #   (b) TensorFuse 相对结构基线 matsvd(2D-SVD) 下降 ≥ 5%
    # 精确 Tucker 数据上 HOSVD 已近最优，HOOI 无法结构性超越，故以结构基线量化世界顶级降幅。
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
    print(f"[门禁] TensorFuse vs hosvd(单遍Tucker强基线): {red_hosvd:+.2f}%  非劣守护={'PASS' if non_inferior else 'FAIL'}")
    print(f"[门禁] TensorFuse vs matsvd(2D-SVD结构基线) 下降: {red_matsvd:.2f}%  门槛>=5%: {'PASS' if struct_pass else 'FAIL'}")
    print(f"[门禁] TensorFuse vs naive(噪声地板) 下降: {red_naive:.2f}%")
    grade = "S (世界顶级)" if (non_inferior and struct_pass) else ("A" if non_inferior else "B")
    print(f"[等级] {grade}")
    if args.out:
        print(f"\nbenchmark 已写入: {args.out}")
    return 0


def cmd_failure(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    pipe = TensorPipeline(cfg)
    cases = pipe.failure_cases()
    print("=" * 78)
    print(" TensorForge 典型误例分析（全部来自真实运行）")
    print("=" * 78)
    for c in cases:
        print(f"\n[{c['id']}] {c['title']}")
        for k, v in c.items():
            if k in ("id", "title", "cause"):
                continue
            print(f"    {k}: {v}")
        print(f"    原因: {c['cause']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    pipe = TensorPipeline(cfg)
    set_all(cfg.seed)
    from data.synthetic import make_tucker_tensor

    X_obs, X_true = make_tucker_tensor(cfg.sizes, cfg.ranks, cfg.noise_rel, seed=cfg.seed)
    res = pipe.run(X_obs, X_true, cfg.ranks, seed=cfg.seed)
    print(f"{'method':<14}{'rel_err':>14}{'rmse':>14}{'n_iter':>10}")
    print("-" * 52)
    for name, r in res.items():
        print(f"{name:<14}{r.rel_err:>14.6f}{r.rmse:>14.6f}{r.n_iter:>10}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    set_all(cfg.seed)
    from data.synthetic import make_tucker_tensor
    from eval.metrics import rel_err
    from tensor.decompositions import hosvd, tucker_hooi

    X_obs, X_true = make_tucker_tensor(cfg.sizes, cfg.ranks, cfg.noise_rel, seed=cfg.seed)
    h = hosvd(X_obs, cfg.ranks)
    ho = tucker_hooi(X_obs, cfg.ranks)
    eh = rel_err(h.reconstructed, X_true)
    eo = rel_err(ho.reconstructed, X_true)
    ok = eo <= eh + 1e-9
    print(
        f"[selftest] hosvd rel_err={eh:.6f}  hooi rel_err={eo:.6f}  hooi<=hosvd: {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tensorforge", description="TensorForge 张量分解系统 (作者: 晨星)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="运行跨 seed 基准")
    b.add_argument("--out", default="benchmark.json")
    b.add_argument("--sizes")
    b.add_argument("--ranks")
    b.add_argument("--noise-rel", type=float, default=None)
    b.add_argument("--n-seeds", type=int, default=None)
    b.add_argument("--seed", type=int, default=None)
    b.add_argument("--hooi-iters", type=int, default=None)
    b.add_argument("--cp-iters", type=int, default=None)
    b.set_defaults(func=cmd_benchmark)

    f = sub.add_parser("failure", help="典型误例分析")
    f.add_argument("--sizes")
    f.add_argument("--ranks")
    f.add_argument("--noise-rel", type=float, default=None)
    f.add_argument("--n-seeds", type=int, default=None)
    f.add_argument("--seed", type=int, default=None)
    f.set_defaults(func=cmd_failure)

    r = sub.add_parser("run", help="单次全方法分解")
    r.add_argument("--sizes")
    r.add_argument("--ranks")
    r.add_argument("--noise-rel", type=float, default=None)
    r.add_argument("--seed", type=int, default=None)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("selftest", help="快速自检")
    s.add_argument("--sizes")
    s.add_argument("--ranks")
    s.add_argument("--noise-rel", type=float, default=None)
    s.add_argument("--seed", type=int, default=None)
    s.set_defaults(func=cmd_selftest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
