"""
timer.py文件提供了时间的相关操作:
1.计时器, 包含到初始时间点和上一时间点的间隔计算
2.时间和字符串互转/获取当前时间/进程休眠的包装, 时间间隔判断

Last update: 2026-02-14 by Junlin_409
version: 1.0.0
"""

# 导入区
import time
from datetime import datetime, timedelta

# 计时器
class Timer:
    def __init__(self) -> None:
        self.time_0 = time.time()
        self.time_i = self.time_0

    def interval_last(self) -> float:
        time_t = self.time_i
        self.time_i = time.time()
        return self.time_i - time_t

    def interval_origin(self) -> float:
        self.time_i = time.time()
        return self.time_i - self.time_0

# (进程)时间延迟
def delay(seconds: float) -> None:
    """
    是time.sleep函数的包装
    """
    time.sleep(seconds)

# 当前时间获取
def current_time() -> datetime:
    """
    是datetime.now函数的包装
    """
    return datetime.now()

# 时间转字符串
def format_time(time_obj: datetime, time_format: str = "%Y%m%d%H%M%S") -> str:
    """
    是datetime.strftime函数的包装
    """
    return datetime.strftime(time_obj, time_format)

# 字符串转时间
def parse_time(time_str: str, time_format: str = "%Y%m%d%H%M%S") -> datetime:
    """
    是datetime.strptime函数的包装
    """
    return datetime.strptime(time_str, time_format)

# 时间间隔判断
def within_the_interval(time_p: datetime, time_n: datetime, delta: dict) -> bool:
    """
    判断两段时间的间隔是否在给定间隔内
    """
    if time_p > time_n:
        time_p, time_n = time_n, time_p
    return time_n - time_p <= timedelta(**delta)
