"""配置：ENV_TENSORFORGE_* 覆盖 + 基础 schema 校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


@dataclass
class Config:
    seed: int = 42
    sizes: tuple = (20, 20, 20)
    ranks: tuple = (3, 3, 3)
    noise_rel: float = 0.12
    n_seeds: int = 5
    hooi_iters: int = 30
    hooi_tol: float = 1e-8
    cp_iters: int = 200
    cp_tol: float = 1e-8
    n_jobs: int = 1

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for f in fields(cls):
            env = "TENSORFORGE_" + f.name.upper()
            if env not in os.environ:
                continue
            raw = os.environ[env]
            cur = getattr(cls, f.name)
            if isinstance(cur, tuple):
                val = tuple(int(x) for x in raw.strip("()").replace(" ", "").split(",") if x != "")
            elif isinstance(cur, bool):
                val = raw.strip().lower() in ("1", "true", "yes", "y")
            elif isinstance(cur, int):
                val = int(raw)
            elif isinstance(cur, float):
                val = float(raw)
            else:
                val = raw
            setattr(cfg, f.name, val)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if len(self.sizes) < 3:
            raise ValueError("[E200] sizes 维度必须 >= 3（本系统聚焦三阶及以上张量）")
        if len(self.ranks) != len(self.sizes):
            raise ValueError("[E200] ranks 长度必须与 sizes 一致")
        for s, r in zip(self.sizes, self.ranks):
            if not (1 <= int(r) <= int(s)):
                raise ValueError(f"[E110] 秩 {r} 超出维度范围 [1,{s}]")
        if not (0.0 < self.noise_rel < 1.0):
            raise ValueError("[E200] noise_rel 必须在 (0,1) 之间")
        if self.n_seeds < 1:
            raise ValueError("[E200] n_seeds 必须 >= 1")

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
