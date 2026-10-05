"""错误码 E100~E500，统一异常基类 ``TensorForgeError``。"""

from __future__ import annotations


class TensorForgeError(Exception):
    code = "E000"
    doc = ""

    def __init__(self, msg: str = "") -> None:
        super().__init__(f"[{self.code}] {msg or self.doc}")


class ShapeMismatchError(TensorForgeError):
    code = "E100"
    doc = "张量形状与模式索引/秩不匹配"


class RankError(TensorForgeError):
    code = "E110"
    doc = "秩配置非法（必须为正整型元组，且不超过各维尺寸）"


class ConvergenceError(TensorForgeError):
    code = "E120"
    doc = "迭代分解未在上限内收敛"


class NotFittedError(TensorForgeError):
    code = "E130"
    doc = "分解器尚未拟合即被查询"


class DataLeakError(TensorForgeError):
    code = "E140"
    doc = "检测到评测信息泄漏（真值参与拟合）"


class ConfigError(TensorForgeError):
    code = "E200"
    doc = "配置项非法或缺失"


# 错误码注册表：撞码/撞名即 RuntimeError（踩坑库 G 节）
_REGISTRY: dict[str, type] = {}


def _register(cls: type) -> type:
    if cls.code in _REGISTRY and _REGISTRY[cls.code] is not cls:
        raise RuntimeError(f"错误码冲突: {cls.code} 已被 {_REGISTRY[cls.code].__name__} 占用")
    _REGISTRY[cls.code] = cls
    return cls


for _c in (
    ShapeMismatchError,
    RankError,
    ConvergenceError,
    NotFittedError,
    DataLeakError,
    ConfigError,
):
    _register(_c)
