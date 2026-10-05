# TensorForge 架构文档

> 作者：晨星 · v0.1.0

本文描述 TensorForge 的模块边界、数据流、关键算法与确定性设计。所有模块纯 numpy 实现，零重型依赖，可离线运行。

---

## 1. 设计目标与原则

| 原则 | 落地 |
|---|---|
| 可量化可复现 | 唯一随机入口 `set_all(seed)`，同 seed 两次运行逐位一致 |
| 干净环境一键复现 | `requirements.lock.txt` 手写可安装版本（`numpy==2.5.3`），非本地 `pip freeze` 路径 |
| 单测 + lint + CI 全绿 | 24 项 pytest + ruff check/format + GitHub Actions 双版本矩阵 |
| 文档齐全 | README / architecture / model_card 齐备，基线表真实来自运行输出 |
| 离线可降级 | 缺失 scipy 自动降级到 numpy 线性代数后端 |
| 无信息泄漏 | 方法选择只用观测张量残差，真值仅用于评测打分 |

---

## 2. 模块拓扑

```
                 ┌─────────────────────────────┐
  synthetic data │  data/synthetic.py          │
  (Tucker + 噪声)│  make_tucker_tensor         │──┐
                 └─────────────────────────────┘  │
                                                  ▼
  ┌───────────────  core/  ────────────────────────────────┐
  │ seed.set_all  · config  · errors(E100-E500) · types ·  │
  │ interfaces(Decomposer Protocol)                         │
  └────────────────────────────────────────────────────────┘
        │                                                      │
        ▼                                                      ▼
  ┌──────────────────────┐                     ┌──────────────────────────┐
  │ preprocess/unfold.py │                     │  tensor/                  │
  │ unfold/fold/ttm      │◄──── used by ──────►│  decompositions.py        │
  │ khatri_rao / kruskal │                     │  naive/matsvd/hosvd/      │
  └──────────────────────┘                     │  tucker_hooi/cp_als       │
                                                │  fuse.py (TensorFuse)     │
   eval/metrics.py ── rel_err/rmse ──────────►  └──────────────────────────┘
                                                │
                                                ▼
                                  ┌──────────────────────────┐
                                  │ pipeline/pipeline.py      │
                                  │ run / benchmark /         │
                                  │ failure_cases / summarize │
                                  └──────────────────────────┘
                                                │
                          ┌──────────────┬──────┴───────┬──────────────┐
                          ▼              ▼              ▼              ▼
                     cli.py        examples/      tests/ (24)    benchmark.json
                  (4 subcmds)      run_demo.py                   (generated)
```

---

## 3. 张量代数基座（`preprocess/unfold.py`）

标准 N 阶张量-矩阵乘（TTM）是整套系统的数学核心：

- `unfold(X, n)`：将第 n 模展平为矩阵 `X_(n)`，shape `(I_n, Π_{k≠n} I_k)`。
- `fold(M, n, shape)`：`unfold` 的逆操作。
- `ttm(X, M, n)`：第 n 模乘矩阵 `M`：`X ×_n M`。
- `khatri_rao_chain(factors)`：多个因子矩阵的 Khatri-Rao 积链，供 CP-ALS 最小二乘更新。
- `kruskal_to_tensor(factors)`：由 CP 因子重建张量。**关键修正**：采用动态字母 einsum 规格（`[a,b,c]r -> abcr`），避免下标含数字导致 `np.einsum` 报错、并防止多维累乘超维。

> 关键不变量测试：HOSVD 重构满足 `X ≈ ttm(ttm(...ttm(G, U0,0)...), U_{N-1}, N-1)`，残差在 1e-12 量级。

---

## 4. 确定性引擎（`core/seed.py`）

唯一入口 `set_all(seed)`：

```python
def set_all(seed: int) -> np.random.Generator:
    global _SEED
    _SEED = int(seed)
    rng = np.random.default_rng(_SEED)
    np.random.seed(_SEED)      # 传统 RandomState
    random.seed(_SEED)          # 标准库 random
    return rng
```

所有随机源（现代 Generator、传统 `np.random`、标准库 `random`）被同一 seed 锁死。CP-ALS 内部初始化使用 `np.random.default_rng(seed)`，保证 bit-for-bit 复现。确定性门禁在 `cli.py benchmark` 与 `examples/run_demo.py` 中执行：同 seed 两次运行 `per_seed` 核心指标逐位比较 = PASS。

---

## 5. 分解算法集（`tensor/decompositions.py`）

| 方法 | 类型 | 说明 |
|---|---|---|
| `naive` | 参考 | 重构 = 观测，代表噪声地板（下界） |
| `matsvd` | 基线 | 仅模式 0 做 2D 截断 SVD，忽略其余维度结构 |
| `hosvd` | 基线 | 高阶 SVD 单遍 Tucker，标准非迭代 |
| `tucker_hooi` | SOTA | Higher-Order Orthogonal Iteration，交替正交迭代逼近最优 Tucker 因子 |
| `cp_als` | SOTA | CP/PARAFAC 交替最小二乘，各模统一秩 |

`DECOMPOSERS` 注册表提供 `available_methods()` / `run_method(name, ...)` 统一调度。所有函数签名 `f(X, ranks, seed=None, **kw) -> DecompResult`。

**HOOI 迭代**：每轮对模式 n 投影 `Z = X ×_k U_kᵀ (k≠n)`，取 `Z_(n)` 前 Rₙ 左奇异向量更新 `U_n`；残差变化 `< tol` 或达 `n_iter` 收敛。

**CP-ALS**：逐模用其余因子的 Khatri-Rao 链做 `np.linalg.lstsq` 最小二乘更新，重建 `kruskal_to_tensor`。

---

## 6. 旗舰融合（`tensor/fuse.py`）

```python
def tensor_fuse(X, ranks, seed=None, n_iter=30, cp_iters=200, tol=1e-8, **kw):
    hooi = tucker_hooi(X, ranks, seed=seed, n_iter=n_iter, tol=tol, **kw)
    cp   = cp_als(X, ranks, seed=seed, n_iter=cp_iters, tol=tol, **kw)
    rh = norm(hooi.rec - X) / norm(X)   # 仅用观测张量
    rc = norm(cp.rec   - X) / norm(X)
    chosen = cp if rc <= rh else hooi
    chosen.name = "tensor_fuse"
    chosen.extra = {"selection": choice, "hooi_resid": rh, "cp_resid": rc, ...}
    return chosen
```

选择准则完全基于**观测张量上的重构残差**（信息无泄漏）。`extra` 记录两路残差与最终选择，供可解释审计。`tensor_fuse_hooi_only` 为固定 HOOI 的消融变体。

---

## 7. 编排层（`pipeline/pipeline.py`）

- `TensorPipeline.run(X_obs, X_true, ranks, seed)`：单张量跑全 6 方法，可选对真值打分。
- `benchmark(seeds, out_path)`：跨 seed 调用 `set_all` → 合成数据 → run → `_aggregate` 聚合 mean±std → 落盘 `benchmark.json`。
- `failure_cases(seed)`：派生 4 个典型误例（过秩/欠秩/高噪/非 Tucker），全部来自真实运行并含原因归因。
- `summarize(report)`：生成可读性能基线表。

**聚合 bug 修复记录**：`_aggregate` 曾误用未定义循环变量 `m_i`（F821），已修正为 `for m_i, m in enumerate(methods)`，并严格按 `rows[m_i]` 索引，保证每方法取自身行。

---

## 8. 评测口径（`eval/metrics.py`）

- `rel_err(X_rec, X_true) = ‖X_rec − X_true‖ / ‖X_true‖`
- `rmse = sqrt(mean((X_rec − X_true)²))`
- `reduction(a, b) = (1 − a/b) × 100%`

门禁用 `reduction` 量化旗舰相对结构基线的降幅。

---

## 9. 配置与错误码

- `core/config.Config`：`TENSORFORGE_*` 环境变量覆盖 + `validate()` 校验（sizes≥3、ranks∈[1,size]、noise_rel∈(0,1)、n_seeds≥1）。
- `core/errors.py`：E100~E500 错误码 + 注册表撞码检测（重复码触发 RuntimeError），确保错误体系自洽。
- `core/interfaces.py`：`Decomposer` Protocol，约束分解函数签名一致性。

---

## 10. CI 与可复现

`.github/workflows/ci.yml`：Ubuntu + Python 3.12/3.13 矩阵，步骤为 `pip install → ruff check/format --check → pytest → python examples/run_demo.py`（demo 内含确定性校验 + 门禁）。

发布走 `scripts/gh_push.py` 三级降级（git push → Git Data API → Contents API），在 git 智能协议被代理拦截时仍可交付。
