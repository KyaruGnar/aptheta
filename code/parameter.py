"""
parameter.py文件归纳了项目中的参数

Last update: 2026-02-15 by Junlin_409
version: 1.0.0 标记
"""

# 引入区
import sys
import torch

# 数值常量
FLOAT_EPSILON = sys.float_info.epsilon
FLOAT_MAXIMUM = sys.float_info.max
FLOAT_MINIMUM = sys.float_info.min

# 路径常量
DATA_PATH = "./data/self-built"
EXTERNAL_PATH = "./external"
MODEL_PATH = "./model"
PYTHON2_PATH = "D:/soft/MGLTools-1.5.6/python"

# JSON字典常量
VINA_WEIGHTS = {
    "Gauss1": -0.035579,
    "Gauss2": -0.005156,
    "Repulsion": 0.840245,
    "Hydrophobic": -0.035069,
    "Hydrogen bonding": -0.587439,
    "Glue": 50,
    "Rot": 0.05846
}

# 半径常量
EXTRA_METAL_COVALENT_RADIUS = 1.75
BEAD_RADIUS = 15.0
BOND_LENGTH_TOLERANCE = 1.1

# 阈值常量
CURL_THRESHOLD = 1000.0

# 机器/深度学习参数常量
SEED = 2048

# 1.种子设置
def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"当前机器学习种子: {SEED}.")

# 2.设备设置
DEVICE = torch.device("cpu")
def set_device(device: str = "auto") -> None:
    global DEVICE # pylint: disable=W0603
    if device == "auto":
        DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    else:
        DEVICE = torch.device(device)
    print(f"当前机器学习设备: {DEVICE}.")
