"""
data_pipeline.py文件获取/转换/解析/流式构造模型的数据集, 是aptheta模型的数据工程:
1.数据获取功能: 从RCSB数据库下载蛋白质文件和对应配体文件
2.
3.
4.数据集构建流程功能: 
EX.(自建)数据存储原则:
(1)根目录: DATA_PATH; 子目录: DATA_PATH/{pdb_id};
(2)蛋白质(所有格式)目录: DATA_PATH/{pdb_id}/protein; 配体(所有格式)目录: DATA_PATH/{pdb_id}/ligands;
(3)对接文件目录: DATA_PATH/{pdb_id}/mlxparam;
"""

# 引入区
import ast
import atexit
import json
import os
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from aptheta_utils import fileio, timer
from extcall import prepare_receptor, prepare_ligand
from parameter import DATA_PATH, SEED, VINA_WEIGHTS

# 常量区
LIGAND_FORMATS = ["sdf", "mol2"]


# 一.数据获取功能
# 本部分提供功能: 从RCSB数据库(1)下载蛋白质文件和对应配体文件; (2)获取蛋白质的ID全集; (3)获取蛋白质的配体;
# 函数统一返回布尔值, True表示下载成功或已有该文件, False表示下载失败
# 0.会话设置
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=0.2,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "POST"]),
    raise_on_status=False
)
session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
atexit.register(session.close)

# 1.根据pdb_id从RCSB数据库下载蛋白质文件
def download_pdb_from_rcsb(pdb_id: str, data_path: str) -> bool:
    os.makedirs(data_path, exist_ok=True)
    pdb_filepath = f"{data_path}/{pdb_id}.pdb"
    if os.path.exists(pdb_filepath):
        return True
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = session.get(url)
    if response.status_code == 200:
        fileio.write_file_text(pdb_filepath, response.text)
        print(f"蛋白质({pdb_id})下载成功.")
        return True
    print(f"蛋白质({pdb_id})下载失败, 错误代码:{response.status_code}.")
    return False

# 2.根据pdb_id和配体位点从RCSB数据库下载蛋白质的配体文件
def download_ligands_from_rcsb(pdb_id: str, protein_info: dict, data_path: str) -> bool:
    os.makedirs(data_path, exist_ok=True)
    for chain_id in protein_info["site_residues"]:
        for res_num, res_name in protein_info["site_residues"][chain_id].items():
            if res_name not in protein_info["bound_ligands"]:
                continue
            for ligand_format in LIGAND_FORMATS:
                ligand = f"{res_name}_{chain_id}_{res_num}"
                ligand_filepath = f"{data_path}/{ligand}.{ligand_format}"
                if os.path.exists(ligand_filepath):
                    continue
                main_url = f"https://models.rcsb.org/v1/{pdb_id}/ligand"
                query_url = f"?auth_asym_id={chain_id}&auth_seq_id={res_num}&encoding={ligand_format}"
                response = session.get(f"{main_url}{query_url}")
                if response.status_code != 200:
                    print(f"蛋白质({pdb_id})配体({ligand})格式({ligand_format})下载失败, 错误代码:{response.status_code}.")
                    return False
                fileio.write_file_text(ligand_filepath, response.text)
                print(f"蛋白质({pdb_id})配体({ligand})格式({ligand_format})下载成功.")
    return True

# 3.从RCSB数据库获取纯蛋白结构的pdb_id全集并转储为文件
def get_pdb_ids_from_rcsb() -> bool:
    main_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    # 只获取蛋白质
    query_json = {
        "query":{
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater",
                        "value": 0
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_nucleic_acid",
                        "operator": "equals",
                        "value": 0
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_nucleic_acid_hybrid",
                        "operator": "equals",
                        "value": 0
                    }
                }
            ]
        },
        "request_options": {"return_all_hits": True},
        "return_type": "entry"
    }
    # 全部结构
    # query_json = {
    #     "query":{"type": "terminal", "service": "text"},
    #     "request_options": {"return_all_hits": True},
    #     "return_type": "entry"
    # }
    pdb_ids = [f"#The creation date: {timer.format_time(timer.current_time())}"]
    response = session.get(f"{main_url}?json={json.dumps(query_json)}")
    if response.status_code == 200:
        entries = response.json()["result_set"]
        pdb_ids.extend(entry["identifier"] for entry in entries)
        print(f"已收集{len(pdb_ids)}个蛋白质的ID.")
        fileio.write_file_lines(f"{DATA_PATH}/pdb_ids.txt", pdb_ids)
        return True
    print(f"蛋白质ID全集获取失败, 错误代码:{response.status_code}.")
    return False

# 4.从RCSB数据库获取指定pdb_id蛋白质的结合配体
def get_bound_ligands_from_rcsb(pdb_id: str, protein_info: dict) -> bool:
    if "bound_ligands" in protein_info:
        return len(protein_info["bound_ligands"]) > 0
    protein_info["bound_ligands"] = []
    main_url = "https://data.rcsb.org/graphql"
    attributes = "{rcsb_entry_info{nonpolymer_bound_components}}"
    query_url = f"{{entry(entry_id: \"{pdb_id}\"){attributes}}}"
    response = session.get(f"{main_url}?query={query_url}")
    if response.status_code == 200:
        bound_ligands: list[str] = response.json()["data"]["entry"]["rcsb_entry_info"]["nonpolymer_bound_components"]
        if bound_ligands is not None:
            protein_info["bound_ligands"] = bound_ligands
            return True
        print(f"蛋白质({pdb_id})配体不存在, 将忽略处理.")
        return False
    print(f"蛋白质({pdb_id})的配体获取失败, 错误代码:{response.status_code}.")
    return False


# 二.数据转换功能
# 1.将蛋白质文件转换为pdbqt形式
def convert_pdb_to_pdbqt(pdb_id: str, data_path: str) -> bool:
    pdb_filepath = f"{data_path}/{pdb_id}_atoms.pdb"
    pdbqt_filepath = f"{data_path}/{pdb_id}.pdbqt"
    if os.path.exists(pdbqt_filepath):
        return True
    return prepare_receptor(pdb_filepath, pdbqt_filepath)

# 2.适配配体的SDF文件并返回原子计数(当前使元素类型正确, 但仍存在电荷校验错误)
def adapt_ligand_sdf(ligand: str, data_path: str) -> int:
    sdf_filepath = f"{data_path}/{ligand}.sdf"
    adapted_sdf_filepath = f"{data_path}/{ligand}_adapted.sdf"
    records = fileio.read_file_lines(sdf_filepath)
    atom_count = int(records[3][0:3].strip())
    if os.path.exists(adapted_sdf_filepath):
        return atom_count
    for index in range(atom_count):
        records[4+index] = records[4+index][0:32] + records[4+index][32].lower() + records[4+index][33:]
    fileio.write_file_lines(adapted_sdf_filepath, records)
    return atom_count

# 3.将配体文件mol2转换为pdbqt形式
def convert_ligands_to_pdbqt(pdb_id: str, protein_info: dict, data_path: str) -> bool:
    try:
        protein_info["prepared_ligands"] = []
        for chain_id in protein_info["site_residues"]:
            for res_num, res_name in protein_info["site_residues"][chain_id].items():
                if res_name not in protein_info["bound_ligands"]:
                    continue
                ligand = f"{res_name}_{chain_id}_{res_num}"
                atom_count = adapt_ligand_sdf(ligand, data_path)
                ligand_filepath = f"{data_path}/{ligand}.mol2"
                pdbqt_filepath = f"{data_path}/{ligand}.pdbqt"
                if os.path.exists(pdbqt_filepath):
                    protein_info["prepared_ligands"].append(ligand)
                    continue
                # 排除金属离子(在adt脚本中无法成键而报错的)影响
                if atom_count > 1 and prepare_ligand(ligand_filepath, pdbqt_filepath):
                    protein_info["prepared_ligands"].append(ligand)
    except Exception as e:
        print(f"Error: {e}")
        return False
    if len(protein_info["prepared_ligands"]) == 0:
        print(f"蛋白质({pdb_id})无可用配体, 将不纳入数据集中")
        return False
    return True


# 三.数据解析功能
# 1.解析蛋白质的PDB文件, 提取出纯坐标的_atoms文件和配体位点信息
def parse_protein_pdb(pdb_id: str, protein_info: dict, data_path: str) -> None:
    pdb_filepath = f"{data_path}/{pdb_id}.pdb"
    pdb_atoms_filepath = f"{data_path}/{pdb_id}_atoms.pdb"
    if os.path.exists(pdb_atoms_filepath) and "site_residues" in protein_info:
        return
    atom_records: list[str] = []
    main_chain: set[str] = set()
    site_residues: dict = {}
    for record in fileio.read_file_lines(pdb_filepath):
        if record.startswith("HET   "):
            res_name = record[7:10].strip()
            chain_id = record[12]
            res_num = int(record[13:17])
            if chain_id not in site_residues:
                site_residues[chain_id] = {}
            site_residues[chain_id][res_num] = res_name
        elif record.startswith("ATOM"):
            atom_records.append(record)
            main_chain.add(record[21])
        elif record.startswith("ENDMDL"):
            break
    protein_info["atom_count"] = len(atom_records)
    protein_info["main_chain"] = sorted(main_chain)
    # 过滤结合位点不在蛋白质链上的残基
    protein_info["site_residues"] = {chain: site for chain, site in site_residues.items() if chain in main_chain}
    fileio.write_file_lines(pdb_atoms_filepath, atom_records)

# 2.从给定pdbqt文件中解析出对接盒子
def parse_box_from_pdbqt(filepaths: list[str], allowance: float = 6.0) -> dict[str, float]:
    min_x = min_y = min_z = 1e-9
    max_x = max_y = max_z = -1e-9
    for filepath in filepaths:
        for record in fileio.read_file_lines(filepath):
            if record.startswith("ATOM"):
                x, y, z = float(record[30:38]), float(record[38:46]), float(record[46:54])
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)
    box = {
        "center_x": (min_x + max_x) / 2,
        "center_y": (min_y + max_y) / 2,
        "center_z": (min_z + max_z) / 2,
        "size_x": max_x - min_x + allowance,
        "size_y": max_y - min_y + allowance,
        "size_z": max_z - min_z + allowance,
    }
    return box

# 3.解析log文件(以当前log格式为准, 16行一个单位, 后续需要同步更改!!!) @Warn
def parse_logfile(log_filename: str) -> list:
    mlxparm_log = fileio.read_file_lines(log_filename)
    raw_dataset = []
    # 临时过滤
    ppids = [pid.lower() for pid in fileio.read_file_lines("./data/pdb_ids.txt")[1:]]
    for log_idx in range(int(len(mlxparm_log)/16)):
        base_idx = log_idx * 16
        # 信息提取
        pdb_id = mlxparm_log[base_idx+1].split("/")[-1].split(".")[0]
        # 临时过滤
        if pdb_id not in ppids:
            continue
        receptor = f"{DATA_PATH}/{pdb_id}/protein/{pdb_id}_atoms.pdb"
        ligand = ast.literal_eval(mlxparm_log[base_idx+3])[0].split("/")[-1].split(".")[0]
        site_chain, site_num = ligand.split("_")[1], int(ligand.split("_")[2])
        ligand = f"{DATA_PATH}/{pdb_id}/ligands/{ligand}_adapted.sdf"
        # 参数修正带来的rmsd指标影响
        xb_rmsds = ast.literal_eval(mlxparm_log[base_idx+5])
        xa_rmsds = ast.literal_eval(mlxparm_log[base_idx+7])
        rmsds = []
        for i, (elem1, elem2) in enumerate(zip(xb_rmsds, xa_rmsds)):
            rmsds.append(elem1)
            rmsds.append(elem2)
        xb_rmsd = ast.literal_eval(mlxparm_log[base_idx+5])[0]
        xa_rmsd = ast.literal_eval(mlxparm_log[base_idx+7])[0]
        # 权重变化值
        weights = []
        for i in range(7):
            name, val_s = mlxparm_log[base_idx+9+i].split(":")
            if name == "Glue":
                continue
            val = float(val_s)
            weights.append((val - VINA_WEIGHTS[name]) / VINA_WEIGHTS[name])
        if site_num > 30:
            raw_dataset.append([pdb_id, receptor, ligand, site_chain, xb_rmsd, xa_rmsd, weights, rmsds])
    return raw_dataset

# 3.1.解析log文件(以当前log格式为准, 16行一个单位, 后续需要同步更改!!!) @Warn
def parse_logfile2(log_filename: str, data_path: str) -> list:
    mlxparm_log = fileio.read_file_lines(log_filename)
    raw_dataset = []
    for log_idx in range(int(len(mlxparm_log)/16)):
        base_idx = log_idx * 16
        # 信息提取
        pdb_id = mlxparm_log[base_idx+1].split("/")[-2]
        receptor = f"{data_path}/PDBbind_v2020_PL/{pdb_id}/{pdb_id}_protein_atoms.pdb"
        ligand = ast.literal_eval(mlxparm_log[base_idx+3])[0].split("/")[-1].split(".")[0]
        ligand = f"{data_path}/PDBbind_v2020_PL/{pdb_id}/ligands/{ligand}_adapted.sdf"
        # 参数修正带来的rmsd指标影响
        xb_rmsds = ast.literal_eval(mlxparm_log[base_idx+5])
        xa_rmsds = ast.literal_eval(mlxparm_log[base_idx+7])
        rmsds = []
        for i, (elem1, elem2) in enumerate(zip(xb_rmsds, xa_rmsds)):
            rmsds.append(elem1)
            rmsds.append(elem2)
        xb_rmsd = ast.literal_eval(mlxparm_log[base_idx+5])[0]
        xa_rmsd = ast.literal_eval(mlxparm_log[base_idx+7])[0]
        # 权重变化值
        weights = []
        for i in range(7):
            name, val_s = mlxparm_log[base_idx+9+i].split(":")
            if name == "Glue":
                continue
            val = float(val_s)
            weights.append((val - VINA_WEIGHTS[name]) / VINA_WEIGHTS[name])
        raw_dataset.append([pdb_id, receptor, ligand, None, xb_rmsd, xa_rmsd, weights, rmsds])
    return raw_dataset


# 四.数据集构建流程类
class DataPipeline:
    def __init__(self, dataset_size: int = 5, data_path: str = DATA_PATH) -> None:
        random.seed(SEED)
        self.dataset_size = dataset_size
        self.dataset: list[str] = []
        self.data_path = data_path

    # 1.获取蛋白质的ID全集
    def get_pdb_ids(self, freshness: int = 90) -> list[str]:
        pdb_ids_filepath = f"{self.data_path}/pdb_ids.txt"
        print("任务: 获取蛋白质ID全集.")
        if os.path.exists(pdb_ids_filepath):
            print("检测到本机已存在数据集, 正在校验时效性.")
            content = fileio.read_file_lines(pdb_ids_filepath)
            file_time = timer.parse_time(content[0][20:])
            if timer.within_the_interval(file_time, timer.current_time(), {"days": freshness}):
                print("已存在较新数据集, 任务结束.")
                return content[1:]
        print("未存在较新数据集, 尝试从RCSB数据库获取蛋白质ID全集.")
        if not get_pdb_ids_from_rcsb():
            return []
        return fileio.read_file_lines(pdb_ids_filepath)[1:]

    # 2.初始化信息集
    def initialize_info(self, pdb_id: str) -> dict:
        protein_info: dict = {
            "pdb_id": pdb_id,
            "protein_path": "",
            "ligands_path": "",
        }
        protein_info["pdb_id"] = pdb_id
        return protein_info

    # 3.准备指定受体
    def prepare_receptor_of(self, pdb_id: str, protein_info: dict) -> bool:
        protein_path = f"{self.data_path}/{pdb_id}/protein"
        os.makedirs(protein_path, exist_ok=True)
        protein_info["protein_path"] = protein_path
        # Step.1 download pdb
        if download_pdb_from_rcsb(pdb_id, protein_path):
            # Step.2 parse pdb
            parse_protein_pdb(pdb_id, protein_info, protein_path)
            # Step.3 convert pdb to pdbqt
            return convert_pdb_to_pdbqt(pdb_id, protein_path)
        return False

    # 4.准备指定受体的配体
    def prepare_ligands_of(self, pdb_id: str, protein_info: dict) -> bool:
        ligands_path = f"{self.data_path}/{pdb_id}/ligands"
        os.makedirs(ligands_path, exist_ok=True)
        protein_info["ligands_path"] = ligands_path
        # Step.1 get bound ligands(功能转移至全流程中)
        # if not get_bound_ligands_from_rcsb(pdb_id, protein_info):
        #     return False
        # Step.2 download ligands
        if download_ligands_from_rcsb(pdb_id, protein_info, ligands_path):
            # Step.3 convert ligand to pdbqt
            if convert_ligands_to_pdbqt(pdb_id, protein_info, ligands_path):
                return True
        return False

    # 5.准备样本(当前项目只使用受体和一个配体作为对接输入)
    def prepare_sample(self, pdb_id: str, precise_pocket: bool = False):
        sample_path = f"{self.data_path}/{pdb_id}"
        protein_info = fileio.read_json(f"{sample_path}/{pdb_id}.json")
        receptor = f"{protein_info["protein_path"]}/{pdb_id}.pdbqt"
        ligands = [f"{protein_info["ligands_path"]}/{random.choice(protein_info["prepared_ligands"])}.pdbqt"]
        input_files = {"receptor": receptor, "ligands": ligands}
        if precise_pocket:
            box = parse_box_from_pdbqt(ligands)
        else:
            box = parse_box_from_pdbqt([receptor] + ligands)
        return sample_path, input_files, box

    # 全流程
    def run(self, pdb_ids: list[str] | None = None, except_ids: list[str] | None = None,
            freshness: int = 90, atoms_constraint: int = 5000):
        """
        EX1.数据存储原则(默认):\n
        (1)根目录: DATA_PATH; 子目录: DATA_PATH/{pdb_id};\n
        (2)蛋白质(所有格式)目录: DATA_PATH/{pdb_id}/protein; 配体(所有格式)目录: DATA_PATH/{pdb_id}/ligands;
        """
        if pdb_ids is None:
            pdb_ids = self.get_pdb_ids(freshness)
            random.shuffle(pdb_ids)
        count = 0
        for pdb_id in pdb_ids:
            pdb_id = pdb_id.lower()
            if except_ids is not None and pdb_id in except_ids:
                continue
            timer.delay(0.3)
            print(f"蛋白质({pdb_id})开始准备, 当前进度({count}/{self.dataset_size})", flush=True)
            os.makedirs(f"{self.data_path}/{pdb_id}", exist_ok=True)
            info_filepath = f"{self.data_path}/{pdb_id}/{pdb_id}.json"
            if os.path.exists(info_filepath):
                protein_info = fileio.read_json(info_filepath)
            else:
                protein_info = self.initialize_info(pdb_id)
            if not get_bound_ligands_from_rcsb(pdb_id, protein_info):
                print(f"蛋白质({pdb_id})无配体, 将不纳入数据集中")
                fileio.write_json(info_filepath, protein_info)
                continue
            if self.prepare_receptor_of(pdb_id, protein_info):
                if protein_info["atom_count"] > atoms_constraint:
                    print(f"蛋白质({pdb_id})过于复杂, 将不纳入数据集中")
                    fileio.write_json(info_filepath, protein_info)
                    continue
                if self.prepare_ligands_of(pdb_id, protein_info):
                    fileio.write_json(info_filepath, protein_info)
                    count += 1
                    self.dataset.append(pdb_id)
                    if count >= self.dataset_size:
                        break
                    continue
            fileio.write_json(info_filepath, protein_info)
            print(f"蛋白质({pdb_id})在受体/配体准备过程中出错, 将不纳入数据集中")
        fileio.write_file_lines(f"{self.data_path}/dataset.txt", self.dataset)

    # 通过位于自建数据集的记录加载数据集
    @classmethod
    def load(cls, dataset_file, data_path="") -> "DataPipeline":
        if not os.path.exists(dataset_file) or not os.path.isfile(dataset_file):
            raise FileNotFoundError("The dataset of self-built is not existed.")
        if data_path == "":
            data_path = os.path.dirname(dataset_file)
        self_built = cls(data_path=data_path)
        self_built.dataset = fileio.read_file_lines(dataset_file)
        if self_built.dataset[-1] == "":
            self_built.dataset.pop()
        return self_built


# 五.PDBBind数据集
class PDBBindDataSet:
    def __init__(self, data_path: str) -> None:
        random.seed(SEED)
        self.data_path = data_path
        self.dataset: list[str] = []

    # 2.准备指定受体
    def prepare_receptor_of(self, pdb_id: str) -> bool:
        pdb_filepath = f"{self.data_path}/{pdb_id}/{pdb_id}_protein.pdb"
        # Step.1 提纯坐标
        pdb_atoms_filepath = f"{self.data_path}/{pdb_id}/{pdb_id}_protein_atoms.pdb"
        if not os.path.exists(pdb_atoms_filepath):
            atom_records: list[str] = []
            for record in fileio.read_file_lines(pdb_filepath):
                if record.startswith("ATOM"):
                    atom_records.append(record)
                elif record.startswith("ENDMDL"):
                    break
            fileio.write_file_lines(pdb_atoms_filepath, atom_records)
        # Step.2 转换格式
        pdbqt_filepath = f"{self.data_path}/{pdb_id}/{pdb_id}_protein.pdbqt"
        if os.path.exists(pdbqt_filepath):
            return True
        return prepare_receptor(pdb_atoms_filepath, pdbqt_filepath)

    # 3.准备指定受体的配体
    def prepare_ligands_of(self, pdb_id: str) -> bool:
        ligand_filepath = f"{self.data_path}/{pdb_id}/{pdb_id}_ligand.mol2"
        pdbqt_filepath = f"{self.data_path}/{pdb_id}/{pdb_id}_ligand.pdbqt"
        if os.path.exists(pdbqt_filepath):
            return True
        return prepare_ligand(ligand_filepath, pdbqt_filepath)

    # 4.准备样本(当前项目只使用受体和一个配体作为对接输入)
    def prepare_sample(self, pdb_id: str, precise_pocket: bool = False):
        sample_path = f"{self.data_path}/{pdb_id}"
        receptor = f"{sample_path}/{pdb_id}_protein.pdbqt"
        ligands = [f"{sample_path}/{pdb_id}_ligand.pdbqt"]
        input_files = {"receptor": receptor, "ligands": ligands}
        if precise_pocket:
            box = parse_box_from_pdbqt(ligands)
        else:
            box = parse_box_from_pdbqt([receptor] + ligands)
        return sample_path, input_files, box

    # 全流程
    def run(self):
        """
        EX3.PDBBind数据存储原则(默认):\n
        (1)根目录: {self.data_path};\n
        (2)子目录: {self.data_path}/{pdb_id}; 蛋白质(所有格式)和配体(所有格式)均在子目录中;\n
        (3)对接文件目录: {self.data_path}/{pdb_id}/mlxparam;
        """
        pdb_ids = [pdb_id for pdb_id in os.listdir(self.data_path)
                   if os.path.isdir(os.path.join(self.data_path, pdb_id))]
        sucess = 0
        failure = 0
        remain = len(pdb_ids)
        for pdb_id in pdb_ids:
            pdb_id = pdb_id.lower()
            print(f"蛋白质({pdb_id})开始准备, 当前进度(成功: {sucess}; 失败: {failure}; 剩余: {remain})", flush=True)
            remain -= 1
            if self.prepare_receptor_of(pdb_id):
                if self.prepare_ligands_of(pdb_id):
                    sucess += 1
                    self.dataset.append(pdb_id)
                    print(f"蛋白质({pdb_id})准备完毕, 纳入数据集.")
                    continue
            failure += 1
            print(f"蛋白质({pdb_id})在受体/配体准备过程中出错, 将不纳入数据集.")
        print(f"完成PDBBind数据集构造, 总和: {len(pdb_ids)}, 成功: {sucess}, 失败: {failure}.")
        print(f"将数据集保存至: {self.data_path}/dataset.txt")
        # dataset_info = [
        #     f"#The creation time: {timer.format_time(timer.current_time())}",
        #     f"#The dataset path: {self.data_path}",
        #     f"#The dataset size: {len(self.dataset)}",
        # ]
        fileio.write_file_lines(f"{self.data_path}/dataset.txt", self.dataset)

    # 通过位于PDBBind的数据集记录加载数据集
    @classmethod
    def load(cls, dataset_file, data_path="") -> "PDBBindDataSet":
        if not os.path.exists(dataset_file) or not os.path.isfile(dataset_file):
            raise FileNotFoundError("The dataset of PDBBind is not existed.")
        if data_path == "":
            data_path = os.path.dirname(dataset_file)
        pdbbind = cls(data_path)
        pdbbind.dataset = fileio.read_file_lines(dataset_file)
        if pdbbind.dataset[-1] == "":
            pdbbind.dataset.pop()
        return pdbbind
