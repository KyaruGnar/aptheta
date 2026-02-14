"""
utility.py文件归纳了实用方法及模型:
1.读写方法: 通用文件读写, 配置加载
2.架构模型: 泛型序列, 泛型映射
3.运算模型: 矩阵, 三角矩阵, 对称矩阵
4.数学模型: 高斯函数, 硬sigmod函数

Last update: 2026-02-14 by Junlin_hpc
version: 1.0.1 (从pkg_resources迁移到importlib.resources)
"""


# 导入区
import json
import math
from importlib import resources
from typing import TypeVar, Generic, Self, Any, cast


# 常量区
T = TypeVar("T") # 任意类型
DEFAULT_VALUES = {
    int: 0,
    float: 0.0,
    str: ""
}


# 一.文件读写方法
# 1.读取文件
def read_file(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return content.split("\n")

# 2.编写文件
def write_file(filepath: str, content: str | list[str], append: bool = False) -> None:
    write_mode = "a" if append else "w"
    with open(filepath, write_mode, encoding="utf-8") as f:
        if isinstance(content, str):
            f.write(content)
        else:
            for line in content:
                f.write(line)
                f.write("\n")

# 3.读取配置文件
def load_configuration(config_name: str) -> dict:
    filepath = resources.files(__package__).joinpath(f"configuration/{config_name}.json")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# 二.架构模型类
# 1.泛型序列
class GenericSequence(Generic[T]):
    @classmethod
    def __class_getitem__(cls, item) -> type[Self]:
        cls.generic_type = item
        return cls

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.generic_type = cls.generic_type

    # 将类变量覆盖为实例变量
    def __init__(self) -> None:
        self.generic_type: type = getattr(self.__class__, "generic_type")

    # 值校验
    # 传入待校验的值,以校验值是否为设定的值类型,返回校验结果.与键校验不同的时,允许值为None.
    def validate_value(self, value: Any) -> bool:
        return isinstance(value, self.generic_type) or value is None

    # 值校正
    # 传入待校正的值,值是否为设定的值类型,返回校验结果.与键校验不同的时,允许值为None.
    # 若校验不通过但值转换控制生效,会尝试将值转换为目标类型,此时值为空则生成默认值.
    def ensure_value(self, value: Any, value_convert: bool = False) -> T | None:
        if self.validate_value(value):
            if value is None and value_convert:
                return cast(T | None, DEFAULT_VALUES.get(self.generic_type, None))
            return value
        if value_convert:
            return self.generic_type(value)
        raise TypeError("The value should meet the type requirement. ")


# 2.泛型映射
K = TypeVar("K") # 任意类型
V = TypeVar("V") # 任意类型
class GenericMapping(Generic[K, V]):
    @classmethod
    def __class_getitem__(cls, item) -> type[Self]:
        cls.generic_type = item
        return cls

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.generic_type = cls.generic_type

    def __init__(self) -> None:
        self.generic_type: tuple[type, type] = getattr(self.__class__, "generic_type")
        self.key_type = self.generic_type[0]
        self.value_type = self.generic_type[1]

    # 键校验
    # 传入待校验的键,以校验值是否为设定的键类型,返回校验结果.
    def validate_key(self, key: Any) -> bool:
        return isinstance(key, self.key_type)

    # 值校验
    # 传入待校验的值,以校验值是否为设定的值类型,返回校验结果.与键校验不同的时,允许值为None.
    def validate_value(self, value: Any) -> bool:
        return isinstance(value, self.value_type) or value is None

    # 值校正
    # 传入待校正的值,值是否为设定的值类型,返回校验结果.与键校验不同的时,允许值为None.
    # 若校验不通过但值转换控制生效,会尝试将值转换为目标类型,此时值为空则生成默认值.
    def ensure_value(self, value: Any, value_convert: bool = False) -> V | None:
        if self.validate_value(value):
            if value is None and value_convert:
                return cast(V | None, DEFAULT_VALUES.get(self.value_type, None))
            return value
        if value_convert:
            return self.value_type(value)
        raise TypeError("The value should meet the type requirement. ")


# 三.运算模型类
# 1.矩阵
class Matrix(GenericSequence[T]):
    def __init__(self, row: int, column: int, default_value: T | None = None, default: bool = True) -> None:
        super().__init__()
        self.row = row
        self.column = column
        self.default = default
        default_value = self.ensure_default_value(default_value)
        self.data = [[default_value] * self.column for _ in range(row)]

    def __getitem__(self, key: tuple[int, int]) -> T | None:
        self.validate_key(key)
        value = self.data[key[0]][key[1]]
        return value

    def __setitem__(self, key: tuple[int, int], value: T | None) -> None:
        self.validate_key(key)
        value = self.ensure_value(value)
        self.data[key[0]][key[1]] = value

    def validate_key(self, key: tuple) -> None:
        if len(key) == 2 and all(isinstance(k, int) for k in key):
            return
        raise TypeError("The key should follow the following format: (i: int, j: int).")

    def ensure_default_value(self, default_value: Any) -> T | None:
        return self.ensure_value(default_value, self.default)

    def expand(self, expanded_row: int, expanded_column: int, default_value: T | None = None) -> None:
        if expanded_row == self.row and expanded_column == self.column:
            return
        if expanded_row < self.row or expanded_column < self.column:
            raise ValueError("The row(column) after expanding should be greater than or equal to the row(column) now.")
        default_value = self.ensure_default_value(default_value)
        for d in self.data:
            d.extend([default_value] * (expanded_column-self.column))
        self.data.extend([[default_value]*expanded_column for _ in range(expanded_row-self.row)])

    def append(self, appended_matrix: "Matrix[T]", default_value: T | None = None) -> None:
        if not isinstance(appended_matrix, Matrix):
            raise TypeError("The input should be an instance of matrix.")
        if appended_matrix.generic_type != self.generic_type:
            raise TypeError("The addition's type should be consistent with the self's.")
        default_value = self.ensure_default_value(default_value)
        self.expand(self.row+appended_matrix.row, self.column+appended_matrix.column, default_value)
        for i in range(appended_matrix.row):
            for j in range(appended_matrix.column):
                self[self.row+i, self.column+j] = appended_matrix[i, j]


# 2.三角矩阵
class TriangularMatrix(Matrix[T]):
    def __init__(self, dimension: int, default_value: T | None = None,
                 default: bool = True, diagonal: bool = True) -> None:
        super().__init__(dimension, dimension, default_value, default)
        self.diagonal = diagonal

    def validate_key(self, key: tuple) -> None:
        super().validate_key(key)
        if key[0] > key[1]:
            return
        if self.diagonal:
            if key[0] == key[1]:
                return
            raise IndexError("The key value should satisfy the condition that i is greater than or equal to j.")
        raise IndexError("The key value should satisfy the condition that i is greater than j.")

    def expand(self, expanded_dimension: int, default_value: T | None = None) -> None: # type: ignore  # pylint: disable=W0221
        super().expand(expanded_dimension, expanded_dimension, default_value)


# 3.对称矩阵
class SymmetricMatrix(Matrix[T]):
    def __init__(self, dimension: int, default_value: T | None = None, default: bool = True) -> None:
        super().__init__(dimension, dimension, default_value, default)

    def __setitem__(self, key: tuple[int, int], value: T | None) -> None:
        super().__setitem__(key, value)
        self.data[key[1]][key[0]] = self.data[key[0]][key[1]]

    def expand(self, expanded_dimension: int, default_value: T | None = None) -> None: # type: ignore  # pylint: disable=W0221
        super().expand(expanded_dimension, expanded_dimension, default_value)


# 四.数学模型类
# 1.高斯函数
# 两种默认初始化: (1)标准正态分布, 需a, b, c参数均为空; (2)正态分布, 需仅a参数为空
class Gaussian:
    def __init__(self, a: float | None = None, b: float | None = None, c: float | None = None) -> None:
        if a is None:
            a = 1.0 / math.sqrt(2*math.pi)
            if b is None and c is None:
                b = 0.0
                c = 1.0
        if b is None or c is None:
            raise ValueError("The values of b and c should both be None if X~N(0,1) else neither.")
        if a <= 0.0 or c <= 0.0:
            raise ValueError("The values of a and c should both be greater than 0.0.")
        self.a = a
        self.b = b
        self.c = c

    def calculate(self, x: float) -> float:
        return self.a * math.exp(-math.pow(x-self.b, 2) / (2*math.pow(self.c, 2)))


# 2.硬sigmod
class HardSigmod:
    def __init__(self, lower_bound: float, upper_bound: float) ->  None:
        if abs(upper_bound - lower_bound) <= 1e-6:
            raise ValueError("The upper bound minus the lower bound should be greater than 1e-6.")
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def calculate(self, x: float) -> float:
        if x < self.lower_bound:
            return 0.0
        if x > self.upper_bound:
            return 1.0
        return (x-self.lower_bound) / (self.upper_bound-self.lower_bound)
