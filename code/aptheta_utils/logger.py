"""
logger.py文件旨在提供全局的日志记录器:
1.一般文件读取/编写操作
2.JSON文件读取/编写操作
3.递归删除文件操作 
"""

# 引入区
import logging
import logging.handlers


class LogEntry:
    def __init__(self, receptor):
        self.receptor = receptor

    def __str__(self):
        return f"Receptor: {self.receptor}," + f"Ligand: {self.receptor}"

# 创建全局日志器
IWEIGHT_LOGGER = logging.getLogger("IWEIGHT")
IWEIGHT_LOGGER.setLevel(logging.DEBUG)

# 控制台输出
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 文件输出（自动轮转）
file_handler = logging.handlers.RotatingFileHandler(
    "app.log",
    maxBytes=100*1024*1024,
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)

 # 统一格式
formatter = logging.Formatter(
    fmt="%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(lineno)d \n %(message)s",
    datefmt="%Y%m%d %H%M%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

IWEIGHT_LOGGER.addHandler(console_handler)
IWEIGHT_LOGGER.addHandler(file_handler)

IWEIGHT_LOGGER.debug("nihao1")
IWEIGHT_LOGGER.info("nihao2")
IWEIGHT_LOGGER.error("nihao3")
IWEIGHT_LOGGER.info(LogEntry("2c0k"))

# FLODER_ = './logs/'
# LOGFILE_ = 'log'
# """
# %(name)s        : 日志记录器的名称
# %(levelno)s     : 打印日志级别的数值。
# %(levelname)s   : 打印日志级别的名称。
# %(pathname)s    : 打印当前执行程序的路径，其实就是sys.argv[0]。
# %(filename)s    : 打印当前执行程序名。
# %(funcName)s    : 打印日志的当前函数。
# %(lineno)d      : 打印日志的当前行号。
# %(asctime)s     : 打印日志的时间。
# %(thread)d      : 打印线程ID。
# %(threadName)s  : 打印线程名称。
# %(process)d     : 打印进程ID。
# %(processName)s : 打印线程名称。
# %(module)s      : 打印模块名称。
# %(message)s     : 打印日志信息。
# """
# # FMT_ = '%(asctime)s %(levelname)8s | %(process)d %(name)s | %(filename)s:%(lineno)d %(message)s'
# FMT_ = '%(asctime)s %(levelname)-8s | %(filename)s:%(lineno)d - %(message)s'
# DATEFMT_ = '%Y-%m-%d %H:%M:%S'


# # 配置文件输出
# ENCODING_ = 'utf-8'
# ### 根据时间切割日志
# # 定义默认日志切割的时间单位，比如 'S'（秒）、'M'（分）、'H'（小时）、'D'（天）等
# WHEN_ = "D"
# # 定义默认日志文件切割的时间间隔，例如当 when='H' 且 interval=1 时，表示每隔一个小时进行一次切割，并生成一个新的日志文件
# INTERVAL_ = 1
# # 定义默认保留旧日志文件的个数（如果超过这个数量，则会自动删除最早的日志文件），默认值为 0，表示不自动删除旧日志文件
# BACKUPCOUNT_ = 0
# ### 根据大小切割日志
# # 定义默认日志文件最大字节数(2M)
# LOG_MAX_BYTES_ = 2 * 1024 * 1024
# # LOG_MAX_BYTES_ = 1024
# # 定义默认日志文件备份个数
# BACKUPCOUNT_ = 5

# class Loggings(object):
#     __instance = None
#     __instance_lock = threading.Lock()

#     def __new__(cls, *args, **kwargs):
#         if not cls.__instance:
#             # 输出日志路径
#             # PATH = os.path.abspath('.') + '/logs/'
#             with Loggings.__instance_lock:
#                 cls.__instance = logging.getLogger()
#                 _formatter = logging.Formatter(fmt=FMT_, datefmt=DATEFMT_)
                
#                 _filename = '{0}{1}_{2}.txt'.format(FLODER_, LOGFILE_, strftime("%Y-%m-%d"))
#                 _file_handler = logging.FileHandler(_filename, encoding=ENCODING_)
#                 _file_handler.setFormatter(_formatter)
#                 cls.__instance.addHandler(_file_handler)

#                 _console_handler = logging.StreamHandler(sys.stdout)
#                 _console_handler.setFormatter(_formatter)
#                 cls.__instance.addHandler(_console_handler)
#                 # 设置日志的默认级别
#                 cls.__instance.setLevel(logging.DEBUG)

#         return cls.__instance

# def init_logging():
#     # 设置日志格式#和时间格式
#     # ansi_format = '\033[95m%(levelname)s:\033[0m%(message)s'  # 95是深紫色背景文字，0是重置颜色
#     # formatter = logging.Formatter(ansi_format)

#     # 拼接日志文件完整路径
#     log_filename = os.path.join(FLODER_, LOGFILE_)
#     # # 使用绝对路径
#     # # 获取当前脚本所在的目录路径。该方法获取不正确时，使用方法二：os.path.realpath(sys.argv[0])
#     # root_path = os.path.dirname(os.path.abspath(sys.argv[0]))
#     # # 如果指定路径不存在，则尝试创建路径
#     # if not os.path.exists(os.path.join(root_path, FLODER_)):
#     #     os.makedirs(os.path.join(root_path, FLODER_))
#     # 使用相对路径
#     if not os.path.exists(FLODER_):
#         os.makedirs(FLODER_)

#     # Create a log format using Log Record attributes
#     # 输出日志路径
#     # PATH = os.path.abspath('.') + '/logs/'
#     _logger = logging.getLogger()
#     _formatter = logging.Formatter(fmt=FMT_, datefmt=DATEFMT_)
    
#     _filename = '{0}.txt'.format(log_filename)
#     # _file_handler = logging.FileHandler(_filename, encoding=ENCODING_)
    
#     ### 根据时间切割日志
#     _file_handler = logging.handlers.TimedRotatingFileHandler(
#                                             filename=_filename,
#                                             when=WHEN_,
#                                             interval=INTERVAL_,
#                                             backupCount=BACKUPCOUNT_,
#                                             encoding=ENCODING_)  # 创建 TimedRotatingFileHandler 实例，即将日志输出到文件的处理器
    
#     # ### 根据大小切割日志
#     # _file_handler = logging.handlers.RotatingFileHandler(
#     #                                         filename=_filename,
#     #                                         maxBytes=LOG_MAX_BYTES_,
#     #                                         backupCount=BACKUPCOUNT_,
#     #                                         encoding=ENCODING_)  # 创建 RotatingFileHandler 实例，即将日志输出到文件的处理器

#     _console_handler = logging.StreamHandler(sys.stdout)

#     _file_handler.setFormatter(_formatter)
#     _console_handler.setFormatter(_formatter)

#     _logger.addHandler(_file_handler)
#     _logger.addHandler(_console_handler)
#     # 设置日志的默认级别
#     _logger.setLevel(logging.DEBUG)

import logging
from logging import StreamHandler

# 创建 Logger
logger = logging.getLogger("level_based_logger")
logger.setLevel(logging.DEBUG)  # 设置Logger为最低级别（确保所有日志都能被处理）

# 1. INFO级别处理器（绿色文字 + 简洁格式）
info_handler = StreamHandler()
info_handler.setLevel(logging.INFO)  # 只处理INFO及以上
info_handler.setFormatter(logging.Formatter(
    "\033[32m%(asctime)s | INFO | %(message)s\033[0m"  # 绿色文字
))

# 2. ERROR级别处理器（红色文字 + 详细格式）
error_handler = StreamHandler()
error_handler.setLevel(logging.ERROR)  # 只处理ERROR及以上
error_handler.setFormatter(logging.Formatter(
    "\033[31m%(asctime)s | ERROR | %(module)s:%(lineno)d | %(message)s\033[0m"  # 红色文字
))

# 添加处理器
logger.addHandler(info_handler)
logger.addHandler(error_handler)

# 测试输出
logger.info("这是一条普通信息")  # 绿色简洁格式
logger.error("这是一条错误信息")  # 红色详细格式