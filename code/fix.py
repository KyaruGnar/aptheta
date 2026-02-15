"""
fix.py文件汇总了项目迭代过程中的修复代码:
1.对旧版本产生结果的修正, 以减少重复执行
2.受限于当前需求的小范围应用

Last update: 2026-02-15 by Junlin_409
version: 1.2.0
"""

# 引入区
import os
import torch
from data_pipeline import PDBBindDataSet
from featurer import extract_sequences_from_pdb
from aptheta_utils import fileio


# 补丁01_20251205(已废除)
# 针对atoms.pdb转pdbqt时格式异常时的探索: 清除H原子
def fix_atoms_hydrogen(atom_records) -> list[str]:
    heavy_atom_records: list[str] = []
    atom_index = 1
    for record in atom_records:
        if record[76:78] == ' H':
            continue
        new_record = f"{record[:6]}{atom_index:>5}{record[11:78]}"
        if len(atom_records) == 80:
            heavy_atom_records.append(f"{new_record}{record[78:]}")
        else:
            heavy_atom_records.append(f"{new_record}  ")
        atom_index += 1
    return heavy_atom_records

# 补丁02_20251211
# 针对RMSD计算时的文件寻找异常和模型不匹配异常的探索: 文件名纠错以及所需文件生成阶段后移
def fix_file(self: PDBBindDataSet):
    """
    补丁02_20251211\n
    针对RMSD计算时的文件寻找异常和模型不匹配异常的探索: 文件名纠错以及所需文件生成阶段后移
    """
    count1, count2 = 0, 0
    for pdb_id in self.dataset:
        # Step.1 atoms.pdb文件名纠错
        old_atoms_filepath = f"{self.data_path}/{pdb_id}/{pdb_id}_atoms.pdb"
        if os.path.exists(old_atoms_filepath):
            new_atoms_filepath = f"{self.data_path}/{pdb_id}/{pdb_id}_protein_atoms.pdb"
            os.rename(old_atoms_filepath, new_atoms_filepath)
            count1 += 1
        # Step.2 ligand_adapted.sdf文件生成后移, 移除原有文件
        adapted_sdf_filepath = f"{self.data_path}/{pdb_id}/{pdb_id}_ligand_adapted.sdf"
        if os.path.exists(adapted_sdf_filepath):
            os.remove(adapted_sdf_filepath)
            count2 += 1
    print(f"补丁02_20251211: 已完成{len(self.dataset)}条数据的规范, 其中文件名纠错{count1}条, 文件移除{count2}条.")

# 补丁03_20260210
# 针对特征预生成过程中的失误进行处理: 统一维度只剩下特征区域而不包含batch_size(1)
def fix_prefeatures(features_filepath: str):
    """
    补丁03_20260210\n
    针对特征预生成过程中的失误进行处理: 统一维度只剩下特征区域而不包含batch_size(1)
    """
    count = 0
    print(f"补丁文件路径: {features_filepath}.")
    for pdb_id in os.listdir(features_filepath):
        try:
            if os.path.isfile(f"{features_filepath}/{pdb_id}"):
                continue
            pstr = torch.load(f"{features_filepath}/{pdb_id}/protein_struct_repr.pth", weights_only=False)
            torch.save(pstr.squeeze(dim=0), f"{features_filepath}/{pdb_id}/protein_struct_repr.pth")
            sr = torch.load(f"{features_filepath}/{pdb_id}/smiles_repr.pth", weights_only=False)
            torch.save(sr.squeeze(dim=0), f"{features_filepath}/{pdb_id}/smiles_repr.pth")
            count += 1
        except Exception as e:
            print(f"{pdb_id}的预权重修复错误: {e}.")
    print(f"补丁03_20260210: 已完成{count}条样本的shape规范.")

# 限制01_20260206
# 数据集筛选(输入格式: pdb_id, pdb_file)
def filter_by_chain_count(self: PDBBindDataSet, count: int = 99999):
    filter_pdb_ids = []
    for pdb_id in self.dataset:
        seqs = extract_sequences_from_pdb(f"{self.data_path}/{pdb_id}/{pdb_id}_protein_atoms.pdb")
        if len(seqs) <= count:
            filter_pdb_ids.append(pdb_id)
    fileio.write_file_lines(f"filter_chain{count}.txt", filter_pdb_ids)
    print(f"限制01_20260208: 已完成从{len(self.dataset)}条数据中筛选出{len(filter_pdb_ids)}条chain数不超过{count}的数据.")
