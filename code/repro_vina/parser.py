"""
parser.py文件归纳了pdbqt文件解析方法:
(1)单文件解析: 从受体/柔性侧链/配体文件中解析出相应模型
(2)多文件解析: 从受体和柔性侧链/配体组/文件组中解析出相应模型
(3)糅合文件解析: 从对接的输出文件中解析出相应模型

Last update: 2026-02-14 by Junlin_hpc
version: 1.0.1 (从pkg_resources迁移到importlib.resources)
"""


# 导入区
import os
from repro_vina.simulation import Atom, Molecule, Flex, Receptor, Ligand, SimpleAtom, SimpleMolecule
from repro_vina.utility import read_file, write_file


# 一.单文件解析
# 1.解析刚性受体(已在Receptor类内实现)
def parse_rigid_receptor_records(records: list[str]) -> Receptor:
    receptor = Receptor()
    for record in records:
        if record.startswith("ATOM"):
            receptor.add_atom(Atom(record))
    return receptor

# 2.解析柔性残基(一个残基文件对应数个残基, BEGIN_RES标志开始, END_RES标志结束, 已在Flex类内实现)
def parse_flex_records(records: list[str]) -> Flex:
    flex = Flex()
    for record in records:
        if record.startswith("ATOM"):
            flex.add_atom(Atom(record))
        elif record.startswith("BRANCH"):
            flex.put_branch((int(record[6:10]), int(record[10:14])))
        elif record.startswith("ENDBRANCH"):
            flex.validate_branch((int(record[9:13]), int(record[13:17])))
        elif record.startswith("BEGIN_RES"):
            flex.anchor_pair = (0, 0)
    return flex

# 3.解析配体(一个配体文件对应一个配体, 已在Ligand类内实现)
def parse_ligand_records(records: list[str]) -> Ligand:
    ligand = Ligand()
    for record in records:
        if record.startswith("ATOM"):
            ligand.add_atom(Atom(record))
        elif record.startswith("BRANCH"):
            ligand.put_branch((int(record[6:10]), int(record[10:14])))
        elif record.startswith("ENDBRANCH"):
            ligand.validate_branch((int(record[9:13]), int(record[13:17])))
        elif record.startswith("TORSDOF"):
            ligand.validate_torsion(int(record[7:10]))
    return ligand

# 4.简易解析(提取出ATOM记录中的序数,坐标和类型)
def parse_records_simply(records: list[str]) -> list[SimpleAtom]:
    simple_atoms: list[SimpleAtom] = []
    for record in records:
        if record.startswith("ATOM"):
            simple_atoms.append(SimpleAtom.from_record(record))
    return simple_atoms

# 二.多文件解析
# 5.解析传入的受体和柔性残基(可选)文件
def parse_receptor_pdbqts(receptor_path: str, flex_path: str | None = None) -> tuple[Receptor, Flex]:
    receptor = Receptor.from_records(read_file(receptor_path))
    flex = Flex.from_records(read_file(flex_path) if flex_path is not None else [])
    return receptor, flex

# 6.解析传入的配体文件集
def parse_ligand_pdbqts(ligand_paths: list[str]) -> list[Ligand]:
    return [Ligand.from_records(read_file(ligand_path)) for ligand_path in ligand_paths]

# 7.简易解析(提取出ATOM记录中的序数,坐标和类型)
def parse_pdbqts_simply(filepaths: list[str]) -> list[SimpleMolecule]:
    simple_models: list[SimpleMolecule] = []
    for filepath in filepaths:
        molecule_type = "ligand" if "ligand" in filepath else "flex"
        simple_models.append(SimpleMolecule(parse_records_simply(read_file(filepath)), molecule_type))
    return simple_models

# 三.糅合(对接结果)文件解析
# 8.分离对接结果文件成单文件
def split_models_record(models_filepath: str, ligands: list[str] | None = None) -> list[list[str]]:
    basename, extension = os.path.splitext(models_filepath)
    molecule_records: list[str] = []
    model_index = 0
    ligand_index = 1
    model_filepaths: list[list[str]] = []
    molecule_filepaths: list[str] = []
    for record in read_file(models_filepath):
        molecule_records.append(record)
        if record.startswith("MODEL"):
            model_index = int(record[6:])
            ligand_index = 1
        elif record.startswith("REMARK INTRA"):
            molecule_records.clear()
        elif record.startswith("TORSDOF"):
            if ligands is None:
                ligand_filepath = f"{basename}_ligand{ligand_index}_{model_index}{extension}"
            else:
                ligand_filepath = f"{basename}_{ligands[ligand_index-1]}_{model_index}{extension}"
            write_file(ligand_filepath, molecule_records[1:])
            molecule_filepaths.append(ligand_filepath)
            molecule_records.clear()
            ligand_index += 1
        elif record.startswith("ENDMDL"):
            if len(molecule_records) > 2:
                flex_filepath = f"{basename}_flex_{model_index}{extension}"
                write_file(flex_filepath, molecule_records[1:-1])
                molecule_filepaths.append(flex_filepath)
            model_filepaths.append(molecule_filepaths)
            molecule_filepaths = []
    return model_filepaths

# 9.解析糅合(对接结果)文件(输出的分子列表中最后一个分子必定为Flex)
def parse_output_pdbqt(output_filename: str) -> list[list[Molecule]]:
    models: list[list[Molecule]] = []
    molecules: list[Molecule] = []
    molecule_records: list[str] = []
    for record in read_file(output_filename):
        molecule_records.append(record)
        if record.startswith("MODEL"):
            molecule_records = []
        elif record.startswith("TORSDOF"):
            molecules.append(Ligand.from_records(molecule_records))
            molecule_records = []
        elif record.startswith("ENDMDL"):
            molecules.append(Flex.from_records(molecule_records))
            models.append(molecules)
            molecules = []
    return models

# 10.简易解析糅合(对接结果)文件
def parse_output_pdbqt_simply(models_filepath: str) -> list[list[SimpleMolecule]]:
    simple_models: list[list[SimpleMolecule]] = []
    simple_molecules: list[SimpleMolecule] = []
    simple_atoms: list[SimpleAtom] = []
    for record in read_file(models_filepath):
        if record.startswith("MODEL"):
            simple_atoms = []
        elif record.startswith("ATOM"):
            simple_atoms.append(SimpleAtom.from_record(record))
        elif record.startswith("TORSDOF"):
            simple_molecules.append(SimpleMolecule(simple_atoms, "ligand"))
            simple_atoms = []
        elif record.startswith("ENDMDL"):
            if len(simple_atoms) > 1:
                simple_molecules.append(SimpleMolecule(simple_atoms, "flex"))
            simple_models.append(simple_molecules)
            simple_molecules = []
    return simple_models
