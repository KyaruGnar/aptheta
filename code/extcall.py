"""
extcall.py归纳了项目使用过程中所用到的外部程序/代码调用:
1.AutoDock区: 调用特定代码文件将蛋白质/配体文件转换为pdbqt格式
2.Vina区: 调用Vina程序生成给定受体和配体的对接结果
3.DockRMSD_YF区: 调用师兄实现的DockRMSD代码文件生成权重优化的评估指标

Last update: 2026-02-15 by Junlin_409
version: 1.0.1 适配linux
"""


# 导入区
import os
import subprocess
import sys
from openbabel import openbabel # type: ignore
from parameter import EXTERNAL_PATH, PYTHON2_PATH, SEED
from aptheta_utils.fileio import read_file_text
from repro_vina.parser import split_models_record

# 常量设置
openbabel.obErrorLog.StopLogging()

# 一.AutoDock区
# 1.调用AutoDock的代码文件将蛋白质文件转换为pdbqt形式
def prepare_receptor(pdb_filepath: str, pdbqt_filepath: str) -> bool:
    """
    input:\n
    pdb_filepath: 转换前的文件路径\n
    pdbqt_filepath: 转换后的文件路径
    """
    call_path = os.path.normpath(f"{EXTERNAL_PATH}/AutoDock/prepare_receptor4.py")
    try:
        subprocess.run(
            [
                PYTHON2_PATH, call_path,
                "-r", pdb_filepath,
                "-o", pdbqt_filepath,
                "-A", "checkhydrogens"
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output}")
        return False

# 2.调用AutoDock的代码文件将配体文件mol2转换为pdbqt形式
def prepare_ligand(ligand_filepath: str, pdbqt_filepath: str) -> bool:
    """
    input:\n
    ligand_filepath: 转换前的文件路径\n
    pdbqt_filepath: 转换后的文件路径
    """
    call_path = os.path.normpath(f"{EXTERNAL_PATH}/AutoDock/prepare_ligand4.py")
    try:
        subprocess.run(
            [
                PYTHON2_PATH, call_path,
                "-l", ligand_filepath,
                "-o", pdbqt_filepath
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output}")
        return False


# 二.Vina区
# 1.调用Vina程序生成对接结果
def vina(input_files: dict, box: dict, output_file: str, weights: dict, cmd_output: bool = False) -> bool:
    """
    input:\n
    input_files: 字典, 包含输入文件receptor, ligands, flex(可能)\n
    box: 字典, 包含对接盒子参数center_(x/y/z), size_(x/y/z)\n
    output_file: 输出文件\n
    weights: 字典, 包含Vina对接参数\n
    cmd_output: 布尔值, 为真则在终端输出
    """
    if sys.platform.startswith("win"):
        vina_bin = "vina.exe"
    else:
        vina_bin = "vina_1.2.5_linux_x86_64"
    vina_path = os.path.normpath(f"{EXTERNAL_PATH}/Vina/{vina_bin}")
    # inputs
    inputs = [
        "--receptor", input_files["receptor"],
        *[cmd for ligand in input_files["ligands"] for cmd in ("--ligand", ligand)],
        *(["--flex", input_files["flex"]] if "flex" in input_files else [])
    ]
    search_area = [
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
    ]
    # outputs
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    outputs = ["--out", output_file]
    # advanced
    advanced = [
        "--weight_gauss1", str(weights["Gauss1"]),
        "--weight_gauss2", str(weights["Gauss2"]),
        "--weight_repulsion", str(weights["Repulsion"]),
        "--weight_hydrophobic", str(weights["Hydrophobic"]),
        "--weight_hydrogen", str(weights["Hydrogen bonding"]),
        "--weight_rot", str(weights["Rot"]),
        "--weight_glue", str(weights["Glue"]),
    ]
    # misc
    misc = [
        "--seed", str(SEED),
        "--cpu", "1",
        "--num_modes", "9",
        "--verbosity", "1"
    ]
    # command
    command = [vina_path] + inputs + search_area + outputs + advanced + misc
    try:
        result = subprocess.run(args=command, stdout=subprocess.PIPE, text=True, check=True)
        if cmd_output:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output}")
        return False


# 三.DockRMSD_YF区
# 0.使用openbabel将文件从pdbqt格式转为sdf格式
def prepare_ligand_sdf(pdbqt_filepath: str, sdf_filepath: str) -> None:
    if os.path.exists(sdf_filepath):
        return
    conv = openbabel.OBConversion()
    mol = openbabel.OBMol()
    conv.SetInAndOutFormats("pdbqt", "sdf")
    conv.ReadString(mol, read_file_text(pdbqt_filepath))
    conv.WriteFile(mol, sdf_filepath)
    conv.CloseOutFile()
    mol.Clear()

# 1.调用师兄提供的代码生成对接的评估指标DockRMSD(input_files中的配体和output_file对应)
def rmsd_eval(input_files: dict, output_file: str) -> list[float]:
    """
    input:\n
    input_files: 字典, 包含对接输入文件receptor, ligands\n
    output_file: 对接输出文件
    """
    computer_path = os.path.normpath(f"{EXTERNAL_PATH}/DockRMSD_YF/ComputeRMSDlig.py")
    ref_rec_path = f"{os.path.splitext(input_files["receptor"])[0]}_atoms.pdb"
    for pdbqt_filepath in input_files["ligands"]:
        sdf_filepath = f"{os.path.splitext(input_files["ligands"][0])[0]}_adapted.sdf"
        prepare_ligand_sdf(pdbqt_filepath, sdf_filepath)
    if len(input_files["ligands"]) > 1:
        ref_lig_path = f"{os.path.dirname(input_files["ligands"][0])}/*_adapted.sdf"
    else:
        ref_lig_path = f"{os.path.splitext(input_files["ligands"][0])[0]}_adapted.sdf"
    rmsds: list[float] = []
    for mol_paths in split_models_record(output_file):
        if "flex" in mol_paths[-1]:
            mol_paths.pop()
        pre_ligs_path_pairs = [(mol_path, f"{os.path.splitext(mol_path)[0]}.sdf") for mol_path in mol_paths]
        for mol_path, pre_lig_path in pre_ligs_path_pairs:
            prepare_ligand_sdf(mol_path, pre_lig_path)
        if len(pre_ligs_path_pairs) > 1:
            pre_lig_path = f"{os.path.dirname(pre_ligs_path_pairs[0][1])}/*.sdf"
        else:
            pre_lig_path = pre_ligs_path_pairs[0][1]
        try:
            result = subprocess.run(
                [
                    "python", computer_path,
                    "--ref_rec_path", ref_rec_path,
                    "--ref_lig_path", ref_lig_path,
                    "--pre_rec_path", ref_rec_path,
                    "--pre_lig_path", pre_lig_path,
                    "--atom_identity_rule", "a"
                ],
                stdout=subprocess.PIPE,
                text=True,
                check=True
            )
            for line in result.stdout.split("\n"):
                if line.startswith("RMSD-lig"):
                    rmsds.append(float(line[11:]))
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.output}")
        for mol_path, pre_lig_path in pre_ligs_path_pairs:
            os.remove(mol_path)
            os.remove(pre_lig_path)
    return rmsds
