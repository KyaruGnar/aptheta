#!/usr/bin/env python3

"""
test文件为各自测试留档
"""
from featurer import extract_sequences_from_pdb, FeatureCollector, generate_protein_struct_repr
from data_pipeline import PDBBindDataSet
from aptheta_utils import fileio
from extcall import prepare_ligand_sdf
import torch
from joblib import Parallel, delayed
from proteinmpnn.protein_mpnn_utils import (gather_nodes, parse_PDB, tied_featurize, # type: ignore
                                            ProteinMPNN, StructureDatasetPDB)
import fix
import os
from parameter import set_seed
from iweight import workflow_bypre
from repro_vina import workflow_with_DockingModel



import sys
import yaml
import re
from pathlib import Path

# import chardet

# # 检测文件编码
# with open("./env/env.yml", 'rb') as f:
#     raw_data = f.read()
#     result = chardet.detect(raw_data)
#     encoding = result['encoding']
#     print(f"检测到编码: {encoding}")

def clean_conda_dep(dep: str) -> str:
    """
    去除 build 编号
    numpy=1.26.4=py310h5f9d8c6_0
    → numpy=1.26.4
    """
    parts = dep.split("=")
    if len(parts) >= 2:
        return "=".join(parts[:2])
    return dep


def split_env_file(env_path: Path):
    with open(env_path, "r", encoding="utf-16") as f:
        env_data = yaml.safe_load(f)

    conda_deps = []
    pip_deps = []

    for dep in env_data.get("dependencies", []):
        if isinstance(dep, str):
            if dep != "pip":
                conda_deps.append(clean_conda_dep(dep))
            else:
                conda_deps.append("pip")

        elif isinstance(dep, dict) and "pip" in dep:
            for p in dep["pip"]:
                pip_deps.append(p.strip())

    env_data.pop("prefix", None)

    new_env_data = {
        "name": env_data.get("name", "env"),
        "channels": env_data.get("channels", []),
        "dependencies": conda_deps,
    }

    conda_output = env_path.with_name(env_path.stem + "_conda.yml")
    pip_output = env_path.with_name("requirements.txt")

    with open(conda_output, "w") as f:
        yaml.dump(new_env_data, f, sort_keys=False)

    if pip_deps:
        with open(pip_output, "w") as f:
            f.write("\n".join(pip_deps))

    # 生成安装脚本

    print(f"✔ Generated: {conda_output}")
    print(f"✔ Generated: {pip_output}")


split_env_file(Path("./env/env.yml"))

# workflow_with_DockingModel(example=0)
# workflow_with_DockingModel(example=1)
# workflow_with_DockingModel(example=2)
# sample_path = "./data/1234"
# print(os.path.split(sample_path))

# pre_log_file = "./log/202602100945"
# pre_fea_file = "./features"
# pdb_ids = []
# for pi in os.listdir(pre_log_file):
#     pdb_ids.append(pi)
# workflow_bypre("sample4500liner3proj512", pdb_ids[:4500], pdb_ids[-100:], pre_log_file, pre_fea_file)

# print(extract_sequences_from_pdb(".\\data\\PDBbind_v2020_PL/3buo/3buo_protein_atoms.pdb"))
# chain_selected = None
# pdb_dict_list = parse_PDB(".\\data\\PDBbind_v2020_PL/3buo/3buo_protein_atoms.pdb")
# dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=200000)
# all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain']
# if chain_selected:
#     designed_chain_list = [str(item) for item in chain_selected.split()]
# else:
#     designed_chain_list = all_chain_list
# fixed_chain_list = [chain_id for chain_id in all_chain_list if chain_id not in designed_chain_list]
# def problematic_task(x):
#     try:
#         if x == 3:
#             raise ValueError(f"故意出错: {x}")
#         else:
#             print(f"{x}!!!!")
#     except Exception as e:
#         print(e)
#     return x * 2

# # verbose=0 不会抑制异常输出！
# try:
#     results = Parallel(n_jobs=2, verbose=0)(
#         delayed(problematic_task)(i) for i in range(5)
#     )
# except Exception as e:
#     print(f"仍然会看到异常: {e}")  # ❌ 异常仍然会输出
# raw_dataset = [('.\\data\\PDBbind_v2020_PL/3bum/3bum_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/1a1c/1a1c_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3bun/3bun_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3buo/3buo_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3buw/3buw_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3bux/3bux_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3bv2/3bv2_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3bv3/3bv3_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3bwf/3bwf_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3bwj/3bwj_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3bx5/3bx5_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3bym/3bym_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3byo/3byo_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3bys/3bys_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3byu/3byu_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3bz3/3bz3_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3bzi/3bzi_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3c0z/3c0z_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3c10/3c10_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3c1k/3c1k_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3c1x/3c1x_protein_atoms.pdb', None), ('.\\data\\PDBbind_v2020_PL/3c39/3c39_protein_atoms.pdb', None)]
# batch_size = 1
# chunks = [
#     generate_protein_struct_repr([(d[0], None) for d in raw_dataset[i:i+batch_size]])
#     for i in range(0, len(raw_dataset), batch_size)
# ]
# bc = torch.cat(chunks, dim=0)
# print(bc)

# dataset = PDBBindDataSet.load(".\\data\\dataset.txt", ".\\data\\PDBbind_v2020_PL")
# fix.filter_by_chain_count(dataset, 1)

# def remove_file():
#     for f in os.listdir("F:\\program\\log3"):
#         if f.endswith("_run"):
#             os.remove(f"F:\\program\\log3\\{f}")
# remove_file()
# fix.fix_prefeatures("./test/features")
# 比较结果
# data_path = "./test/features/1g4j"
# for pth in os.listdir(data_path):
#     a = torch.load(f"{data_path}/{pth}")
#     b = torch.load(f"{data_path}_2/{pth}")
#     print(a, b)
#     print(pth, torch.eq(a, b))
# 预特征流程
# set_seed()
# dataset_file = ".\\only_one_chain_dataset.txt"
# data_path = ".\\data\\PDBbind_v2020_PL"
# dataset = fileio.read_file_lines(dataset_file)
# if dataset[-1] == "":
#     dataset.pop()
# raw_dataset = []
# print(len(dataset))
# for d in dataset:
#     # prepare_ligand_sdf(f"{data_path}/{d}/{d}_ligand.pdbqt", f"{data_path}/{d}/{d}_ligand_adapted.sdf")
#     raw_dataset.append([d, f"{data_path}/{d}/{d}_protein_atoms.pdb", f"{data_path}/{d}/{d}_ligand_adapted.sdf", None])
# print("finish dataing")
# fc = FeatureCollector()
# fc.pre_generate(raw_dataset, "./features")

# fc.generate(raw_dataset, 2)
# fc.save("one_chain_features.pth")


# import matplotlib.pyplot as plt

# layers = [3, 4, 3]  # 输入 / 隐藏 / 输出
# x_gap = 2

# fig, ax = plt.subplots()

# for i, n in enumerate(layers):
#     y = range(n)
#     x = [i * x_gap] * n
#     ax.scatter(x, y, s=600)

#     if i > 0:
#         for y1 in range(layers[i-1]):
#             for y2 in range(n):
#                 ax.plot([(i-1)*x_gap, i*x_gap], [y1, y2], lw=0.6)

# ax.axis('off')
# plt.show()
# from data_pipeline import PDBBindDataSet
# ds = PDBBindDataSet.load("./data/PDBbind_v2020_PL/dataset.txt")
# print(ds.dataset[-1] =="")
# from mlxparam import MlXParamTrainer
# from data_pipeline import PDBBindDataSet

# def process_one_pdb(pdb_id):
#     ps = PDBBindDataSet("F:\\program\\test\\data1")
#     mpt = MlXParamTrainer(10, 3)
#     mpt.setup_model()
#     res = mpt.outer_train(*ps.prepare_sample(pdb_id))
#     return res  # 返回值随意
# print(process_one_pdb("1eed"))
# print(process_one_pdb("1eed"))
# from featurer import FeatureCollector
# f = FeatureCollector()
# f.generate([["1cea", "F:\\program\\test\\data0\\1cea\\1cea_atoms.pdb", "F:\\program\\test\\data0\\1cea\\1cea_ligand_adapted.sdf", None]], 1)
# print(3*0.25)

# 20251120 测试
# import os
# from data_pipeline import parse_logfile
# xiba
# dp = "F:\PDBbind\\2020\\PDBbind_v2020_PL"
# pdb_ids = set([pdb_id for pdb_id in os.listdir(dp)
#                    if os.path.isdir(os.path.join(dp, pdb_id))])
# zjps = set([r[0] for r in parse_logfile("./log/20251023010518/result.txt")])
# print(pdb_ids & zjps)
# stdres = ['CYS', 'ILE', 'SER', 'VAL', 'GLN', 'LYS', 'ASN',
#           'PRO', 'THR', 'PHE', 'ALA', 'HIS', 'GLY', 'ASP',
#           'LEU', 'ARG', 'TRP', 'GLU', 'TYR', 'MET', 'HID',
#           'HSP', 'HIE', 'HIP', 'CYX', 'CSS']
# res = ("THR PRO ALA PHE ASN LYS PRO LYS VAL GLU LEU HIS VAL HIS LEU ASP GLY ALA ILE LYS PRO GLU THR ILE LEU TYR "
#     "PHE GLY LYS LYS ARG GLY ILE ALA LEU PRO ALA ASP THR VAL GLU GLU LEU ARG ASN ILE ILE GLY MET ASP LYS PRO "
#     "LEU SER LEU PRO GLY PHE LEU ALA LYS PHE ASP TYR TYR MET PRO VAL ILE ALA GLY CYS ARG GLU ALA ILE LYS ARG "
#     "ILE ALA TYR GLU PHE VAL GLU MET LYS ALA LYS GLU GLY VAL VAL TYR VAL GLU VAL ARG TYR SER PRO HIS LEU LEU "
#     "ALA ASN SER LYS VAL ASP PRO MET PRO TRP ASN GLN THR GLU GLY ASP VAL THR PRO ASP ASP VAL VAL ASP LEU VAL "
#     "ASN GLN GLY LEU GLN GLU GLY GLU GLN ALA PHE GLY ILE LYS VAL ARG SER ILE LEU CYS CYS MET ARG HIS GLN PRO "
#     "SER TRP SER LEU GLU VAL LEU GLU LEU CYS LYS LYS TYR ASN GLN LYS THR VAL VAL ALA MET ASP LEU ALA GLY ASP "
#     "GLU THR ILE GLU GLY SER SER LEU PHE PRO GLY HIS VAL GLU ALA TYR GLU GLY ALA VAL LYS ASN GLY ILE HIS ARG "
#     "THR VAL HIS ALA GLY GLU VAL GLY SER PRO GLU VAL VAL ARG GLU ALA VAL ASP ILE LEU LYS THR GLU ARG VAL GLY "
#     "HIS GLY TYR HIS THR ILE GLU ASP GLU ALA LEU TYR ASN ARG LEU LEU LYS GLU ASN MET HIS PHE GLU VAL CYS PRO "
#     "TRP SER SER TYR LEU THR GLY ALA TRP ASP PRO LYS THR THR HIS ALA VAL VAL ARG PHE LYS ASN ASP LYS ALA ASN "
#     "TYR SER LEU ASN THR ASP ASP PRO LEU ILE PHE LYS SER THR LEU ASP THR ASP TYR GLN MET THR LYS LYS ASP MET "
#     "GLY PHE THR GLU GLU GLU PHE LYS ARG LEU ASN ILE ASN ALA ALA LYS SER SER PHE LEU PRO GLU GLU GLU LYS LYS "
#     "GLU LEU LEU GLU ARG LEU TYR ARG GLU TYR GLN")
# ress = set(res.split(" "))
# print(ress)
# print(ress - set(stdres))

# from data_pipeline import PDBBindDataSet, DataPipeline
# pbd = PDBBindDataSet("./test/data0")
# pbd.run()

# import os
# path = "test"
# items = os.listdir(path)
# subdir_names = [item for item in items if os.path.isdir(os.path.join(path, item))]
# print(subdir_names)
# seqs = {"A": "c1", "B": "c2", "C": "c3", "D": "999"}
# chains = "ABDAE"
# # print(":".join([seqs.get(chain) for chain in chains]))
# print(":".join(seqs.values()))

# 20251119 运行多次的结果比较
# from data_pipeline import parse_logfile
# from parameter import set_seed
# from iweight import ProteinDataset, timer, torch
# import random

# logfile = "./log/20251023010518/result.txt"
# set_seed()

# # 1.创建初始数据集
# raw_dataset = parse_logfile(logfile)
# print(f"数据集大小: {len(raw_dataset)}")
# datasetA = ProteinDataset(raw_dataset)
# print("已完成一份")
# timer.delay(1)
# datasetB = ProteinDataset(raw_dataset)
# for i in [12, 45]:
#     da = datasetA.__getitem__(i)
#     print(i, da)

# 调用外部脚本进行分子3D可视化测试
# a = './data/1fda/mlxparam/out10.pdbqt'
# import shutil
# from data_pipeline import parse_logfile
# from repro_vina.parser import split_models_record
# from aptheta_utils import remove_files, write_json
# import os
# from openbabel import openbabel
# def generate_vis_sample(pdb_id, best_epoch):
#     lrs = parse_logfile("./log/20251023010518/result.txt")
#     target = None
#     for lr in lrs:
#         if lr[0] == pdb_id:
#             target = lr
#             break
#     print(target)
#     shutil.copyfile(target[1], "F:\\test\\receptor.pdb")
#     shutil.copyfile(target[2], "F:\\test\\ligand.sdf")
#     output = f'./data/{pdb_id}/mlxparam/out0.pdbqt'
#     remove_files("F:\\test\\docking_results")
#     os.makedirs("F:\\test\\docking_results", exist_ok=True)
#     shutil.copyfile(output, "F:\\test\\output.pdbqt")
#     conv = openbabel.OBConversion()
#     mol = openbabel.OBMol()
#     for i, pre_ligs in enumerate(split_models_record("F:\\test\\output.pdbqt")):
#         pre_ligs_ = []
#         for pre_lig in pre_ligs:
#             conv.ReadFile(mol, pre_lig)
#             result_lig_ = f"F:\\test\\docking_results\\result_{i+1}_0.sdf"
#             conv.WriteFile(mol, result_lig_)
#     output = f'./data/{pdb_id}/mlxparam/out{best_epoch}.pdbqt'
#     shutil.copyfile(output, "F:\\test\\output.pdbqt")
#     for i, pre_ligs in enumerate(split_models_record("F:\\test\\output.pdbqt")):
#         pre_ligs_ = []
#         for pre_lig in pre_ligs:
#             conv.ReadFile(mol, pre_lig)
#             result_lig_ = f"F:\\test\\docking_results\\result_{i+1}_1.sdf"
#             conv.WriteFile(mol, result_lig_)
#     # [15.94069, 16.89697, 15.92684, 15.89473, 15.94925, 16.83945, 17.30487, 15.85512, 17.25884]
#     tmp_dict = {
#         "epoch": best_epoch,
#         "protein": target[1],
#         "ligand": target[2],
#         "rmsds": target[7]
#     }
#     write_json("F:\\test\\info.json", tmp_dict)
# generate_vis_sample("1fda", 10)


# from featurer import generate_protein_seq_reprs, extract_sequences_from_pdb
# s = generate_protein_seq_reprs([("3vmm", extract_sequences_from_pdb("./data/3vmm/protein/3vmm_atoms.pdb").get("A"))])
# print(s)
# from parameter import VINA_WEIGHTS
# weights = [v for v in VINA_WEIGHTS.values()]
# print(weights)
# 20251031
# 多线程学习
# import os

# # 获取CPU核心数
# cpu_count = os.cpu_count()
# print(f"CPU核心数: {cpu_count}")

# import multiprocessing

# # 获取CPU核心数
# cpu_count = multiprocessing.cpu_count()
# print(f"CPU核心数: {cpu_count}")

# import psutil  #  pip install psutil

# # 获取逻辑CPU数量
# logical_cpu_count = psutil.cpu_count(logical=True)
# # 获取物理CPU数量
# physical_cpu_count = psutil.cpu_count(logical=False)

# print(f"逻辑CPU数量: {logical_cpu_count}")
# print(f"物理CPU数量: {physical_cpu_count}")

# # 获取当前系统中所有进程的线程数量
# def get_thread_count():
#     total_threads = 0
#     for proc in psutil.process_iter(['pid', 'name', 'num_threads']):
#         try:
#             total_threads += proc.info['num_threads']
#         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
#             pass
#     return total_threads

# threads = get_thread_count()
# print(f"当前系统中总共的线程数量: {threads}")


# import time
# import threading

# def task1():
#     print("任务1开始")
#     time.sleep(2)  # 模拟耗时操作
#     print("任务1结束")

# def task2():
#     print("任务2开始")
#     time.sleep(2)  # 模拟耗时操作
#     print("任务2结束")

# def main():
#     # 创建线程
#     thread1 = threading.Thread(target=task1)
#     thread2 = threading.Thread(target=task2)

#     # 启动线程
#     thread1.start()
#     thread2.start()

#     # 等待线程完成
#     thread1.join()
#     thread2.join()

# if __name__ == "__main__":
#     main()


# from data_pipeline import download_pdb_from_rcsb, get_pdb_ids_from_rcsb
# get_pdb_ids_from_rcsb()
# download_pdb_from_rcsb("4hhb", ".")
# 20251027: CPU和GPU进行权重学习的时间对比
# import json
# from aptheta_utils import fileio, timer
# from data_pipeline import DataPipeline, parse_logfile
# from featurer import MlXParamTrainer
# from parameter import DATA_PATH, set_seed, set_device
# from iweight import workflow
# import os
# set_seed()
# set_device("cpu")
# dp = DataPipeline(dataset_size=200)
# dp.run(["3vmm"], freshness=900)
# mlxp = MlXParamTrainer()
# test_timer = timer.Timer()
# mlxp.setup_model(weight_decay=0.0)
# for d in dp.dataset:
#     sp, ifs, b = dp.prepare_sample(d, precise_pocket=True)
#     mlxp.outer_train(sp, ifs, b)
#     cpu_time = test_timer.interval_last()
#     set_device()
#     mlxp.outer_train(sp, ifs, b)
#     print(f"CPU用时: {cpu_time}")
#     print(f"GPU用时: {test_timer.interval_last()}")
# 20251024: 异常样本可视化(unused)
# import shutil
# import os
# from aptheta_utils.fileio import read_file_text, remove_files, write_json
# from openbabel import openbabel # type: ignore
# from repro_vina.parser import split_models_record
# from extcall import rmsd_eval
# ref_rec_path = "./data/1fda/protein/1fda_atoms.pdb"
# ref_lig_path = "./data/1fda/ligands/SF4_A_107_adapted.sdf"
# output_file = ".\\data\\1fda\\mlxparam\\out10.pdbqt"
# ifs = {"receptor": ref_rec_path, "ligands": [ref_lig_path]}
# shutil.copyfile(ref_rec_path, "F:\\test\\receptor.pdb")
# shutil.copyfile(ref_lig_path, "F:\\test\\ligand.sdf")
# remove_files("F:\\test\\docking_results")
# os.makedirs("F:\\test\\docking_results", exist_ok=True)
# for i, mol_paths in enumerate(split_models_record(output_file)):
#     if "flex" in mol_paths[-1]:
#         mol_paths.pop()
#     pre_ligs_path_pairs = [(mol_path, f"{os.path.splitext(mol_path)[0]}.sdf") for mol_path in mol_paths]
#     for mol_path, pre_lig_path in pre_ligs_path_pairs:
#         conv = openbabel.OBConversion()
#         mol = openbabel.OBMol()
#         conv.SetInAndOutFormats("pdbqt", "sdf")
#         conv.ReadString(mol, read_file_text(mol_path))
#         result_lig_ = f"F:\\test\\docking_results\\result_{i+1}.sdf"
#         conv.WriteFile(mol, result_lig_)
#         conv.CloseOutFile()
#         mol.Clear()
#     tmp_dict = {
#         "epoch": "best",
#         "protein": "1FDA",
#         "ligand": "SF4_A_107",
#         "rmsds": f"{rmsd_eval(ifs, output_file)}"
#     }
#     write_json("F:\\test\\info.json", tmp_dict)

# from featurer import extract_sequences_from_pdb
# print(extract_sequences_from_pdb('./data/1cnw/protein/1cnw_atoms.pdb'))
# print(extract_sequences_from_pdb('./data/3dig/protein/3dig_atoms.pdb'))
# 20251020: 兼容测试
# from extcall import rmsd_eval

# print(rmsd_eval({"receptor": "./data/1z4m/protein/1z4m.pdbqt", "ligands": ["./data/1z4m/ligands/U5P_A_1001.pdbqt"]},
#           "./data/1z4m/mlxparam/out0.pdbqt"))
# [1.54104, 1.62208, 2.43397, 2.23763, 6.14011, 5.22814, 4.87622, 5.24517, 4.87341]
# [1.54104, 1.62208, 2.43397, 2.23763, 6.14011, 5.22814, 4.87622, 5.24517, 4.87341]
# 20251020: 索引测试
# import random
# random.seed(2048)
# print(random.choice(["a", "b", "c"]))
# print(random.choice(["aassassssssssssssssssssssssssssssssssa", "dadssab", "ccasad"]))
# 20251020: 解包测试
# a = {"b": 3, "c": 5}
# def pp(b, c):
#     print(f"b={b}; c={c}")
# pp(**a)

# 20251009: pandas测试
# import numpy as np
# import pandas as pd
# df = pd.DataFrame(columns=['Name', 'Age', 'City'], dtype=str)
# print(df.dtypes)
# # 特征列（字符串）
# feature = np.array(["C", "H", "O", "N", "C", "T"])

# # 映射表
# mapping = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007}

# # 使用 pandas 实现映射
# result = pd.Series(feature).map(mapping).to_numpy()

# print(result)
# 输出: array([12.011,  1.008, 15.999, 14.007, 12.011,  1.008])

# 2050930: dtype测试
# from repro_vina.simulation import Atom
# import numpy as np
# a = Atom("ATOM      0  ??? ??? ?   1      12.345  -3.000 -10.987  0.00  0.00     0.000 ? ")
# nc = np.array([a.coordinate])
# print(nc.dtype)

# 20250929: vina测试
# from extcall import vina
# from data_pipeline import parse_box_from_pdbqt
# from parameter import VINA_WEIGHTS
# input_files = {
#     "receptor": "F:\\program\\source_data\\2r.pdbqt",
#     "ligands": ["F:\\program\\source_data\\2c0k_C_HEM.pdbqt", "F:\\program\\source_data\\2c0k_D_OXY.pdbqt"],
#     "flex": "F:\\program\\source_data\\2f.pdbqt"
# }
# box = parse_box_from_pdbqt(list(input_files["ligands"]))
# vina(input_files, box, "F:\\program\\source_data\\outtest.pdbqt", VINA_WEIGHTS, True)

# 20250928： pyllint测试
# pylint: disable=C0103
# A_c = 3
# # pylint: enable=C0103
# A_c = 3

# 20250927: conv测试
# import os
# import psutil
# import gc
# def is_file_locked(filepath):
#     """检查文件是否被进程占用"""
#     try:
#         for proc in psutil.process_iter():
#             try:
#                 files = proc.open_files()
#                 for file in files:
#                     if file.path == os.path.abspath(filepath):
#                         return True, proc.name()
#             except (psutil.AccessDenied, psutil.NoSuchProcess):
#                 continue
#         return False, None
#     except ImportError:
#         return "未知（需要psutil库）", None
# from openbabel import openbabel # type: ignore
# fp = 'F:\\program\\source_data\\2c0k\\mlxform\\out0_ligand1_1.pdbqt'
# fp1 = 'F:\\program\\source_data\\2c0k\\mlxform\\out0_ligand1_2.pdbqt'
# conv = openbabel.OBConversion()
# mol = openbabel.OBMol()
# conv.ReadFile(mol, fp)
# conv.CloseOutFile()
# del conv
# gc.collect()
# print(is_file_locked(fp))
# conv.ReadFile(mol, fp1)
# print(is_file_locked(fp))
# 20250926：list测试
# files = {
#     "receptor": "a",
#     "ligands": ["b", "c"],
#     # "flex": "d"
# }
# inputs = [
#     "--receptor", files["receptor"],
#     *[item for tup in [("--ligand", ligand) for ligand in files["ligands"]] for item in tup],
#     *(["--flex", files["flex"]] if files["flex"] else [])
# ]
# print(inputs)
# 20250926：os.path测试
# import os
# a = os.path.dirname("./data/3vmm/ligands/ADP_A_503.pdbqt")
# print(a)
# 20250926：glob测试
# import glob
# a = [i for i in glob.glob("./data/3vmm/ligands/*_adapted.sdf")]
# print(a)
# 20250926：rmsd测试
# from extcall import rmsd_eval
# print(rmsd_eval(input_files={"receptor": "F:\\program\\source_data\\2c0k\\2c0k.pdbqt",
#                        "ligands": ["F:\\program\\source_data\\2c0k\\ligands\\HEM_A1152.pdbqt"]},
#         output_file="F:\\program\\source_data\\2c0k\\mlxform\\out0.pdbqt"))

# 20250924: subprocess测试
# import subprocess
# subprocess.run([
#     "python", "main.py", "3"
# ])

# 20250923: 划分文件测试
# from repro_vina import parser
# print(parser.split_models_record("./real_test/9m4w2l2f.pdbqt"))
# import numpy as np
# a = np.array([1, 2, 3])
# for ii, w in enumerate(a):
#     print(ii, w)
# 20250919： 绘图测试
# import matplotlib.pyplot as plt
# import numpy as np
# # 生成示例数据
# np.random.seed(42)
# n_points = 15
# a_values = np.random.uniform(-2, 6, n_points)
# b_values = np.random.uniform(-3, 8, n_points)
# names = [f'数据点_{i}' for i in range(n_points)]

# # 创建图形
# fig, ax = plt.subplots(figsize=(12, 8))
# scatter = ax.scatter(a_values, b_values, alpha=0.7, c='blue', edgecolors='w', s=100)

# # 设置坐标系
# ax.spines['left'].set_position('zero')
# ax.spines['bottom'].set_position('zero')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlim(-2, 6)
# ax.set_ylim(min(b_values) - 1, max(b_values) + 1)
# ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
# ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
# ax.set_xlabel('X轴', fontsize=12)
# ax.set_ylabel('Y轴', fontsize=12)
# ax.set_title('点击点显示名称', fontsize=14)
# ax.grid(True, linestyle='--', alpha=0.3)

# # 添加注释
# annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
#                     bbox=dict(boxstyle="round", fc="w"),
#                     arrowprops=dict(arrowstyle="->"))
# annot.set_visible(False)

# # 点击事件显示名称
# def onclick(event):
#     if event.inaxes == ax:
#         cont, ind = scatter.contains(event)
#         if cont:
#             idx = ind['ind'][0]
#             annot.xy = (a_values[idx], b_values[idx])
#             annot.set_text(f"{names[idx]}\n({a_values[idx]:.2f}, {b_values[idx]:.2f})")
#             annot.set_visible(True)
#             fig.canvas.draw_idle()
#         else:
#             annot.set_visible(False)
#             fig.canvas.draw_idle()

# fig.canvas.mpl_connect("button_press_event", onclick)

# plt.tight_layout()
# plt.show()

# 20250918：测试gpu
# import torch
# import subprocess
# import os

# def check_gpu_status():
#     print("=" * 50)
#     print("GPU和CU状态检查")
#     print("=" * 50)
    
#     # 1. 检查PyTorch CUDA支持
#     print("1. PyTorch CUDA支持:")
#     print(f"   CUDA available: {torch.cuda.is_available()}")
#     print(f"   CUDA version: {torch.version.cuda}")
#     print(f"   Number of GPUs: {torch.cuda.device_count()}")
    
#     # 2. 检查NVIDIA驱动
#     print("\n2. NVIDIA驱动检查:")
#     try:
#         nvidia_smi = subprocess.check_output(['nvidia-smi'], text=True)
#         print("   NVIDIA驱动正常")
#         # 提取GPU信息
#         lines = nvidia_smi.split('\n')
#         for line in lines:
#             if 'GPU' in line and 'Name' in line:
#                 print(f"   {line.strip()}")
#     except (subprocess.CalledProcessError, FileNotFoundError):
#         print("   NVIDIA驱动未安装或nvidia-smi不可用")
    
#     # 3. 检查CUDA工具包
#     print("\n3. CUDA工具包检查:")
#     try:
#         nvcc_version = subprocess.check_output(['nvcc', '--version'], text=True)
#         print("   CUDA工具包已安装")
#         print(f"   {nvcc_version.splitlines()[0]}")
#     except (subprocess.CalledProcessError, FileNotFoundError):
#         print("   CUDA工具包未安装或nvcc不可用")
    
#     # 4. 检查环境变量
#     print("\n4. 环境变量检查:")
#     print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', '未设置')}")
#     print(f"   PATH中包含CUDA: {'cuda' in os.environ.get('PATH', '').lower()}")
    
#     # 5. 尝试手动设置设备
#     print("\n5. 设备设置测试:")
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"   Torch设备: {device}")
    
#     return torch.cuda.is_available()

# # 运行检查
# if check_gpu_status():
#     print("\n✅ GPU可用，可以开始使用了！")
# else:
#     print("\n❌ GPU不可用，请根据上述提示排查问题")

# from rdkit import Chem
# from CompuateRMSD import PointBasedBestRMSD
# query = "test2.mol2"
# template = "2c0k_C_HEM.mol2"
# mol1 = Chem.RemoveAllHs(Chem.MolFromMol2File(query,sanitize=False),sanitize=False)
# mol2 = Chem.RemoveAllHs(Chem.MolFromMol2File(template,sanitize=False),sanitize=False)
# alignedrmsd,rmsd,_ = PointBasedBestRMSD(mol1,mol2)
# print(f"Alignedrmsd: {alignedrmsd}. Rmsd: {rmsd}.")
# from openbabel import openbabel

# conv = openbabel.OBConversion()
# conv.SetInAndOutFormats("pdbqt", "mol2")
# # conv.SetInAndOutFormats("sdf", "mol2")

# mol = openbabel.OBMol()
# # conv.ReadFile(mol, "hem.sdf")
# conv.ReadFile(mol, "out31222.pdbqt")
# conv.WriteFile(mol, "test2.mol2")
# query = "test2.mol2"
# # command = f"DockRMSD {query} {template}"
# command = f"python .\\CompuateRMSD.py --input .\\{query} --ref .\\{template}"
# print(f"DockRMSD command: {command}")
# os.system(command)

# from rdkit import Chem
# pth = os.path.join(".", template)
# print(pth)
# mol1 = Chem.RemoveAllHs(Chem.MolFromMol2File(pth,sanitize=False))

# cmd_output = True
# EXTERNAL_FILE_LOCATION = ".\\external\\"
# data = "9m4w2l2f.pdbqt"
# command = f"{EXTERNAL_FILE_LOCATION}vina_split --input {EXTERNAL_FILE_LOCATION}{data}"
# if not cmd_output:
#     command = f"{command} > nul 2>&1"
# print(f"vina command: {command}")
# os.system(command)


# class Tee:
#     def __init__(self, filename):
#         self.file = open(filename, 'w')
#         self.stdout = sys.stdout

#     def write(self, message):
#         self.file.write(message)
#         self.stdout.write(message)

#     def flush(self):
#         self.file.flush()
#         self.stdout.flush()

# sys.stdout = Tee('output.txt')
# c = (3, 41111)
# a, b = c
# print(a, b, c)
# import subprocess
# import sys
# from proteinmpnn import protein_mpnn_run
# protein_mpnn_run.main()
# # 方法1.1: 使用subprocess调用
# cmd = [
#     sys.executable, "-c", 
#     "import some_library; some_library.main()",
#     "--batch-size", "32",
#     "--lr", "0.001",
#     "--weight-path", "./data/weights.pth"
# ]
# subprocess.run(cmd)
# import importlib
# original_argv = sys.argv.copy()
# from parameter import DATA_PATH
# pdb_id = "3vmm"
# # 设置新的命令行参数
# sys.argv = [
#     "protein_mpnn_run.py", 
#     "--pdb-path", f"{DATA_PATH}/{pdb_id}/protein/{pdb_id}_atoms.pdb",
#     "--pdb-path-chains", "A",
#     "--out-folder", "abcdefg",
#     "--num-seq-per-target", "2",
#     "--sampling-temp", "0.1",
#     "--seed" , "37",
#     "--batch-size", "1"
# ]

# try:
#     from proteinmpnn import protein_mpnn_run
#     importlib.reload(protein_mpnn_run)
#     protein_mpnn_run.main()
# finally:
#     # 恢复原始argv
#     sys.argv = original_argv
# import numpy as np
# from numpy import array
# from sklearn.metrics import mean_squared_error, r2_score
# all_targets = [array([-0.21553725,  1.7012607 , -0.03626521, -0.        , -0.05855887,
#                       0.        ], dtype=np.float32)]
# all_preds = [array([ 4.155722  ,  1.5558367 ,  2.1353414 , -0.529473  , -0.869719  ,
#                    -0.57785034], dtype=np.float32)]
# r2_score(all_targets, all_preds)
# print(r2_score(all_targets, all_preds, multioutput=""))
# print(np.var(all_targets))
# print(mean_squared_error(all_targets,all_preds))
# print(1- mean_squared_error(all_targets,all_preds)/ np.var(all_targets))