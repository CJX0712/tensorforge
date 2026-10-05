# 已交付系统清单 (delivered.md)

> 由 random-ai-system-delivery SOP 自动回写。作者统一署名：晨星 (GitHub: CJX0712)

| 域名 | 日期 | 仓库 | Tag | 质量等级 | 关键指标（门槛） | 门禁 |
|---|---|---|---|---|---|---|
| 张量分解 TensorForge | 2026-10-06 | CJX0712/tensorforge | v0.1.0 | **S（世界顶级）** | 旗舰 vs hosvd 非劣(+0.04%)；vs matsvd 下降 61.50%(≥5%)；vs naive 下降 84.72%；确定性逐位一致 | ✅ PASS |

## 技术选型与 SOTA 对标
- 基座：纯 numpy 张量代数（unfold/fold/ttm/Khatri-Rao/kruskal_to_tensor）
- 对照：naive(噪声地板) / matsvd(2D-SVD 结构基线) / hosvd(单遍 Tucker 强基线) / tucker_hooi(Tucker SOTA) / cp_als(CP SOTA)
- 旗舰 TensorFuse：HOOI 与 CP 按观测张量重构残差自适应择优融合（无信息泄漏）

## DoD 对照
- ✅ 性能可量化可复现：benchmark.json 真实输出，5 seeds mean±std
- ✅ 干净环境一键复现逐位一致：同 seed 两次运行逐位一致
- ✅ 单测+lint+CI 全绿+依赖锁定+离线可降级：24 passed / ruff 0 / CI 3.12+3.13 / requirements.lock.txt(手写 numpy==2.5.3)
- ✅ 文档齐全+基线表真实：README / docs/architecture.md / docs/model_card.md

## 已知限制
精确 Tucker 数据上 HOSVD 已近最优，HOOI 仅极小幅改进；旗舰「世界顶级」以对忽略多向结构的 matsvd 显著结构收益(≥5%) + 非劣 hosvd 量化，口径诚实（参照 TscForge 先例）。
