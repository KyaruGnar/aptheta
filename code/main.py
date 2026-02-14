"""
main.py文件是aptheta模型的流程集合:
1.PDBBind数据集战备

"""

# import time
import sys
# from reader import parse_receptor_pdbqts, parse_ligand_pdbqts, parse_output_pdbqt_simply
# from main_model import DockingModel, DockingModelWithPoses
# from data_processor import DataPipeline, parse_logfile
# from iweight_utils import fileio
# # from ml_xform import set_seed
import json
from aptheta_utils import fileio, timer
from data_pipeline import DataPipeline, parse_logfile
# from featurer import MlXParamTrainer
from parameter import DATA_PATH, set_seed, set_device
from iweight import workflow
import os

class Tee:
    def __init__(self, filename):
        self.file = open(filename, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

from data_pipeline import PDBBindDataSet
def generate_pdbbind_dataset():
    pbd = PDBBindDataSet("F:\\PDBbind\\2020\\PDBbind_v2020_PL")
    pbd.run()


def main():
    set_seed()
    # log_path = f"./log/{timer.format_time(timer.current_time())}"
    # os.makedirs(log_path, exist_ok=True)
    # sys.stdout = Tee(f"{log_path}/log.txt")
    # dp = DataPipeline(dataset_size=200)
    # 屏蔽运行:
    # 1isc: 莫名卡死, 但是调试能完成流程
    # 3aic: 原子数过大存在报错
    # dp.run(except_ids=["1isc", "3aic"], freshness=900)
    # mlxp = MlXParamTrainer()
    # mlxp.setup_model(weight_decay=0.0)
    # for d in dp.dataset:
    #     try:
    #         sp, ifs, b = dp.prepare_sample(d, precise_pocket=True)
    #         mlxp.record(sp, ifs, b, f"{log_path}/result.txt")
    #     except Exception as e:
    #         print(f"pdb {d}训练过程中出错，此样本作废.")
    raw_set = parse_logfile("./log/20251023010518/result.txt")
    
    print(raw_set)
    # workflow(f"{DATA_PATH}/log.txt")
    # sys.stdout = sys.stdout.stdout

    

# 以DockingModel为模型的流程
# def main_in_docking_model():
#     model = DockingModel()

#     # 解析文件
#     time_t = time.time()
#     file_location = "./source_data/"
#     receptor_filename = file_location + "2r.pdbqt"
#     flex_filename = file_location + "2f.pdbqt"
#     model.set_receptor(*parse_receptor_pdbqts(receptor_filename, flex_filename))
#     print(f"解析受体用时: {time.time()-time_t}s.", flush=True)
#     time_t = time.time()
#     ligand_filenames = ["2c0k_C_HEM.pdbqt", "2c0k_D_OXY.pdbqt"]
#     model.set_ligands(parse_ligand_pdbqts(file_location+ligand for ligand in ligand_filenames))
#     print(f"解析配体用时: {time.time()-time_t}s.", flush=True)

#     # 设置运算类进行运算
#     time_t = time.time()
#     energies = model.vina()
#     print(f"最终结合能:{energies[0]:7.3f}, 分子内能量:{energies[1]:6.3f}, 分子间能量:{energies[2]:7.3f}.")
#     print(f"分数运算用时: {time.time()-time_t}s.", flush=True)

#     # 流程结束
#     print("流程结束.")


# # 以MultiplePosesModel为模型的流程
# def main_in_multiple_poses_model():

#     # 受体设置
#     file_location = "./source_data/"
#     receptor_filename = file_location + "2c0k1.pdbqt"
#     ligand_filename = file_location + "2c0k_C_HEM.pdbqt"
#     # receptor_filename = file_location + "2r.pdbqt"
#     time_t = time.time()
#     model = DockingModelWithPoses()
#     model.set_receptor(*parse_receptor_pdbqts(receptor_filename))
#     model.set_ligands(parse_ligand_pdbqts([ligand_filename]))
#     print(f"解析受体文件用时: {time.time()-time_t}s.", flush=True)

#     # 实际计算设置
#     time_t = time.time()
#     output_filename = file_location + "test_out.pdbqt"
#     # output_filename = file_location + "9m4w2l2f.pdbqt"
    
#     model.set_poses(parse_output_pdbqt_simply(output_filename))
#     print(f"解析输出模型用时: {time.time()-time_t}s.", flush=True)

#     # 输出基本信息

#     # 设置运算类进行运算
#     time_t = time.time()
#     model.vina_with_poses(print_mode=True)
#     print(f"计算用时: {time.time()-time_t}s.", flush=True)

#     # 流程结束
#     print("流程结束.")

import math
def parse_log(filename):
    ss = fileio.read_file_lines(filename)
    pids = []
    res = {}
    rest = []
    for i in range(int(len(ss)/16)):
        ori_rmsds = json.loads(ss[i*16+5])
        best_rmsds = eval(ss[i*16+7])
        if ori_rmsds[0] > best_rmsds[0]:
            pids.append((ss[i*16+1], ss[i*16+3], ori_rmsds[0]-best_rmsds[0]))
            idx = math.log10(ori_rmsds[0]-best_rmsds[0])
            res[math.floor(idx)] = res.get(math.floor(idx), 0) + 1
            rest.append(ori_rmsds[0]-best_rmsds[0])
    print(f"已运行数据集个数：{int(len(ss)/16)}")
    print(f"改良数据集个数：{len(pids)}")
    print(f"改良数据集：{pids}")
    print(res)
    rest.sort()
    print(rest)

if __name__ == "__main__":
    generate_pdbbind_dataset()
