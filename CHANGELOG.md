# Changelog

## v0.1.0 (2026-10-06)
- 首发：张量分解系统 TensorForge
- 算法：naive / matsvd(2D SVD 基线) / hosvd(单遍 Tucker) / tucker_hooi / cp_als / 旗舰 TensorFuse
- 数学基座：N 阶通用 unfold / fold / ttm / Khatri-Rao / kruskal / tucker 重建
- 确定性：default_rng(seed) 全局锁死，同 seed 两次运行逐位一致
- 评测：重构相对误差 rel_err / rmse；跨 5 seed 报 mean±std
- 交付：单测 + ruff + CI(3.12/3.13) + 离线 demo + 失败案例分析
- 作者：晨星
