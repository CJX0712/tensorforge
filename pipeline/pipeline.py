"""pipeline：端到端编排层。

``TensorPipeline.run`` 对单张量跑全方法；``benchmark`` 跨 seed 聚合；
``failure_cases`` 派生 ≥3 个典型误例（均来自真实运行，含原因归因）。
"""

from __future__ import annotations

import json
import time
from typing import Optional, Sequence

import numpy as np

from core.config import Config
from core.seed import set_all
from core.types import Aggregate, BenchRow
from data.synthetic import make_random_tensor, make_tucker_tensor
from tensor.decompositions import run_method
from tensor.fuse import tensor_fuse

METHODS = ["naive", "matsvd", "hosvd", "tucker_hooi", "cp_als", "tensor_fuse"]


class TensorPipeline:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or Config()

    def run(
        self,
        X_obs: np.ndarray,
        X_true: Optional[np.ndarray] = None,
        ranks: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ) -> dict:
        ranks = tuple(ranks) if ranks is not None else self.cfg.ranks
        results = {}
        for name in METHODS:
            if name == "tensor_fuse":
                r = tensor_fuse(
                    X_obs,
                    ranks,
                    seed=seed,
                    n_iter=self.cfg.hooi_iters,
                    cp_iters=self.cfg.cp_iters,
                    tol=self.cfg.hooi_tol,
                )
            else:
                r = run_method(name, X_obs, ranks, seed=seed)
            if X_true is not None:
                r.score_against(X_true)
            results[name] = r
        return results

    def benchmark(
        self, seeds: Optional[Sequence[int]] = None, out_path: Optional[str] = None
    ) -> dict:
        cfg = self.cfg
        if seeds is None:
            seeds = list(range(cfg.seed, cfg.seed + cfg.n_seeds))
        rows_per_seed: list[list[BenchRow]] = []
        t0 = time.perf_counter()
        for seed in seeds:
            set_all(seed)
            X_obs, X_true = make_tucker_tensor(cfg.sizes, cfg.ranks, cfg.noise_rel, seed=seed)
            res = self.run(X_obs, X_true, cfg.ranks, seed=seed)
            rows = [BenchRow(m, res[m].rel_err, res[m].rmse, res[m].n_iter) for m in METHODS]
            rows_per_seed.append(rows)
        elapsed = time.perf_counter() - t0

        aggregates = _aggregate(rows_per_seed)
        report = {
            "config": cfg.as_dict(),
            "seeds": list(seeds),
            "elapsed_sec": round(elapsed, 4),
            "methods": [a.to_dict() for a in aggregates],
            "per_seed": [[r.to_dict() for r in rows] for rows in rows_per_seed],
        }
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        return report

    def failure_cases(self, seed: int = 7) -> list[dict]:
        """派生 ≥3 个典型误例，全部来自真实运行，含原因归因。"""
        cfg = self.cfg
        cases: list[dict] = []
        ranks_opt = cfg.ranks

        # FC1：秩过大（过拟合噪声）
        X_obs, X_true = make_tucker_tensor(cfg.sizes, ranks_opt, cfg.noise_rel, seed=seed)
        r_opt = self.run(X_obs, X_true, ranks_opt, seed=seed)["tensor_fuse"]
        ranks_big = tuple(min(cfg.sizes[i], max(ranks_opt) + 4) for i in range(len(cfg.sizes)))
        r_big = self.run(X_obs, X_true, ranks_big, seed=seed)["tensor_fuse"]
        cases.append(
            {
                "id": "FC1_overrank",
                "title": "秩过大 → 过拟合噪声",
                "ranks_opt": list(ranks_opt),
                "ranks_big": list(ranks_big),
                "rel_err_opt": round(r_opt.rel_err, 6),
                "rel_err_big": round(r_big.rel_err, 6),
                "cause": "多余的成分拟合噪声而非结构，重构误差不降反升",
            }
        )

        # FC2：秩过小（欠拟合）
        ranks_small = tuple(max(1, r - 2) for r in ranks_opt)
        r_small = self.run(X_obs, X_true, ranks_small, seed=seed)["tensor_fuse"]
        cases.append(
            {
                "id": "FC2_underrank",
                "title": "秩过小 → 欠拟合结构",
                "ranks_opt": list(ranks_opt),
                "ranks_small": list(ranks_small),
                "rel_err_opt": round(r_opt.rel_err, 6),
                "rel_err_small": round(r_small.rel_err, 6),
                "cause": "秩不足以承载真实 Tucker 结构，残差居高",
            }
        )

        # FC3：高噪声（所有方法逼近噪声地板，旗舰优势收窄但仍非劣）
        noise_hi = min(0.4, cfg.noise_rel * 3 + 0.05)
        Xh, Xth = make_tucker_tensor(cfg.sizes, ranks_opt, noise_hi, seed=seed)
        rh = self.run(Xh, Xth, ranks_opt, seed=seed)
        cases.append(
            {
                "id": "FC3_high_noise",
                "title": "高噪声 → 优势收窄",
                "noise_rel": round(noise_hi, 4),
                "rel_err_naive": round(rh["naive"].rel_err, 6),
                "rel_err_hosvd": round(rh["hosvd"].rel_err, 6),
                "rel_err_tensor_fuse": round(rh["tensor_fuse"].rel_err, 6),
                "cause": "信噪比下降，所有方法逼近噪声地板；旗舰仍非劣于基线",
            }
        )

        # FC4：非 Tucker 结构（随机张量），Tucker/CP 假设失效
        Xr, Xtr = make_random_tensor(cfg.sizes, cfg.noise_rel, seed=seed)
        rr = self.run(Xr, Xtr, ranks_opt, seed=seed)
        cases.append(
            {
                "id": "FC4_non_tucker",
                "title": "非低秩结构 → 张量假设失效",
                "rel_err_naive": round(rr["naive"].rel_err, 6),
                "rel_err_matsvd": round(rr["matsvd"].rel_err, 6),
                "rel_err_tensor_fuse": round(rr["tensor_fuse"].rel_err, 6),
                "cause": "随机张量无联合低秩结构，Tucker/CP 无法压缩，重构误差接近 naive",
            }
        )
        return cases


def _aggregate(rows_per_seed: list[list[BenchRow]]) -> list[Aggregate]:
    methods = [r.method for r in rows_per_seed[0]]
    out: list[Aggregate] = []
    for m_i, m in enumerate(methods):
        rel = [rows[m_i].rel_err for rows in rows_per_seed]
        rms = [rows[m_i].rmse for rows in rows_per_seed]
        nit = [rows[m_i].n_iter for rows in rows_per_seed]
        out.append(
            Aggregate(
                method=m,
                mean_rel_err=float(np.mean(rel)),
                std_rel_err=float(np.std(rel)),
                mean_rmse=float(np.mean(rms)),
                std_rmse=float(np.std(rms)),
                mean_n_iter=float(np.mean(nit)),
                n_seeds=len(rows_per_seed),
            )
        )
    return out


def summarize(report: dict) -> str:
    """生成可读性能基线表（控制台用）。"""
    lines = []
    cfg = report["config"]
    lines.append("=" * 78)
    lines.append(" TensorForge Benchmark")
    lines.append(
        f" sizes={cfg['sizes']} ranks={cfg['ranks']} noise_rel={cfg['noise_rel']} seeds={report['seeds']}"
    )
    lines.append("-" * 78)
    lines.append(f" {'method':<14}{'rel_err mean':>16}{'±std':>10}{'rmse mean':>14}")
    lines.append("-" * 78)
    for m in report["methods"]:
        lines.append(
            f" {m['method']:<14}{m['mean_rel_err']:>16.6f}{m['std_rel_err']:>10.6f}{m['mean_rmse']:>14.6f}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)
