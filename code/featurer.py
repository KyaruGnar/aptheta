"""
featurer.py文件通过原始数据和预训练模型生成模型的特征输入, 是aptheta模型的特征工程:
1.蛋白质结构特征(由ProteinMPNN模型生成)
2.蛋白质序列特征(由esm2模型生成)
3.小分子SMILES特征(由ChemBERTa模型生成)
更新日志:
# The newest update time: 202602082313
# The update device: 409
"""

# 引入区
import importlib.resources as pkg_resources
import os
import torch
from transformers import AutoTokenizer, AutoModel
import esm  # type: ignore
from proteinmpnn.data import vanilla_model_weights # type: ignore
from proteinmpnn.protein_mpnn_utils import (gather_nodes, parse_PDB, tied_featurize, # type: ignore
                                            ProteinMPNN, StructureDatasetPDB)
from Bio import PDB
from rdkit import Chem, RDLogger
from aptheta_utils import timer, fileio
from parameter import DEVICE, EXTERNAL_PATH

# 库设置
RDLogger.DisableLog('rdApp.warning')


# 一.蛋白质结构特征
# 1.ProteinMPNN的编码器的表征提取
# pylint: disable=C0103
class ProteinMPNNEncoder(ProteinMPNN):
    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, # pylint: disable=R0914,R0917
                use_input_decoding_order=False, decoding_order=None):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        return h_V
# pylint: enable=C0103

# 2.特征生成(by ProteinMPNNEncoder模型)
# data格式: [(pdb1, chain_selected1), (pdb2, chain_selected2), ...]
# repr形状: [len(data), 128]
# PS: 模型参数均是库调用的数值指定化
def generate_protein_struct_repr(data, device: torch.device = DEVICE):
    try:
        with pkg_resources.path(vanilla_model_weights, "v_48_020.pt") as weight_path:
            state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model = ProteinMPNNEncoder(num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                                   num_encoder_layers=3, num_decoder_layers=3,
                                   augment_eps=0.0, k_neighbors=state_dict["num_edges"]).to(device)
        model.load_state_dict(state_dict["model_state_dict"])
        model.eval()
    except Exception as e:
        print(e)
    struct_reprs = []
    try:
        for pdb, chain_selected in data:
            pdb_dict_list = parse_PDB(pdb)
            dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=200000)
            all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain']
            if chain_selected:
                designed_chain_list = [str(item) for item in chain_selected.split()]
            else:
                designed_chain_list = all_chain_list
            fixed_chain_list = [chain_id for chain_id in all_chain_list if chain_id not in designed_chain_list]
            chain_id_dict = {}
            chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)
            with torch.no_grad():
                for _, protein in enumerate(dataset_valid):
                    feature = tied_featurize([protein], device, chain_id_dict)
                    # 实际参数: repr = model(X, mask, residue_idx, chain_encoding_all)
                    randn_1 = torch.randn(feature[3].shape, device=feature[0].device)
                    encoder_repr = model(feature[0], feature[1], feature[2], feature[4]*feature[10],
                                        feature[12], feature[5], randn_1)
                    mask = feature[2].unsqueeze(-1).float()
                    sum_repr = torch.sum(encoder_repr * mask.squeeze(0), 1)
                    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
                    struct_reprs.append(sum_repr / sum_mask)
        return torch.cat(struct_reprs, dim=0)
    except Exception as e:
        print(e)
        print(data)


# 二.蛋白质序列特征
# 1.特征提取(from PDB文件)
def extract_sequences_from_pdb(pdb_file: str) -> dict[str]:
    parser = PDB.PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    ppb = PDB.PPBuilder()
    seqs = {}
    for model in structure:
        for chain in model:
            seq = ""
            for pp in ppb.build_peptides(chain):
                seq += str(pp.get_sequence())
            if seq:
                seqs[chain.id] = seq
    return seqs

# 2.特征加工(by esm2_t33_650M_UR50D模型)
# data格式: [(label1, seq1), (label2, seq2), ...]
# repr形状: [len(data), 1280]
def generate_protein_seq_reprs(data, device: torch.device = DEVICE):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    _, _, batch_tokens = batch_converter(data)  # 0=begin, 2=end, 1=padding
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        outputs = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_reprs = outputs["representations"][33]
    seq_reprs = []
    for i, tokens_len in enumerate(batch_lens):
        seq_reprs.append(token_reprs[i, 1 : tokens_len - 1].mean(0))
    return seq_reprs


# 三.小分子SMILES特征
# 1.特征提取(from SDF文件)
def extract_smiles_from_sdf(sdf_file: str) -> str:
    mol = Chem.MolFromMolFile(sdf_file) # pylint: disable=E1101
    if mol is not None:
        smiles = Chem.MolToSmiles(mol) # pylint: disable=E1101
        return smiles
    return ""

# 2.特征加工(by ChemBERTa-77M-MTR模型)
# data格式: [smiles1, smiles2, ...]
# repr形状: [len(data), 384]
def generate_smiles_reprs(data, device: torch.device = DEVICE):
    tokenizer = AutoTokenizer.from_pretrained(f"{EXTERNAL_PATH}/ChemBERTa-77M-MTR")
    encoded_inputs = tokenizer(data, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    model = AutoModel.from_pretrained(f"{EXTERNAL_PATH}/ChemBERTa-77M-MTR").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    attention_mask = encoded_inputs["attention_mask"].unsqueeze(-1).float()
    sum_reprs = torch.sum(outputs.last_hidden_state * attention_mask, 1)
    sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
    smiles_reprs = sum_reprs / sum_mask
    return smiles_reprs


# 四.特征收集器
class FeatureCollector:
    def __init__(self):
        self.features = {}

    # 常规生成, 结合特征工程使用
    def generate(self, raw_dataset, batch_size: int = 16) -> None:
        tm = timer.Timer()
        chunks = [
            generate_protein_struct_repr([(d[1], None) for d in raw_dataset[i:i+batch_size]])
            for i in range(0, len(raw_dataset), batch_size)
        ]
        self.features["vs_struct"] = torch.cat(chunks, dim=0)
        print(f"生成结构特征用时: {tm.interval_last()}s. 所用设备: {DEVICE.type}", flush=True)
        chunks = [
            generate_protein_seq_reprs([(d[0], filter_sequences_by_chains(extract_sequences_from_pdb(d[1]), d[3]))
                                        for d in raw_dataset[i:i+batch_size]])
            for i in range(0, len(raw_dataset), batch_size)
        ]
        t_chunks = [t for c in chunks for t in c]
        self.features["vs_seq"] = torch.stack(t_chunks, dim=0)
        print(f"生成序列特征用时: {tm.interval_last()}s. 所用设备: {DEVICE.type}", flush=True)
        chunks = [
            generate_smiles_reprs([extract_smiles_from_sdf(d[2]) for d in raw_dataset[i:i+batch_size]])
            for i in range(0, len(raw_dataset), batch_size)
        ]
        self.features["vs_smiles"] = torch.cat(chunks, dim=0)
        print(f"生成smiles特征用时: {tm.interval_last()}s. 所用设备: {DEVICE.type}", flush=True)

    def save(self, filepath: str) -> None:
        torch.save(self.features, filepath)

    def load(self, filepath: str) -> None:
        self.features = torch.load(filepath)

    # 样本生成, 预特征存储
    def pre_generate(self, raw_dataset, filepath: str) -> None:
        os.makedirs(filepath, exist_ok=True)
        log_filepath = "{filepath}/abormality.txt"
        abnormal_samples = 0
        for data in raw_dataset:
            try:
                os.makedirs(f"{filepath}/{data[0]}", exist_ok=True)
                if not os.path.exists(f"{filepath}/{data[0]}/protein_struct_repr.pth"):
                    pstr = generate_protein_struct_repr([(data[1], None)])
                    torch.save(pstr.squeeze(dim=0), f"{filepath}/{data[0]}/protein_struct_repr.pth")
                if not os.path.exists(f"{filepath}/{data[0]}/protein_seq_repr.pth"):
                    pser = generate_protein_seq_reprs([(data[0],
                                                        filter_sequences_by_chains(extract_sequences_from_pdb(data[1]),
                                                                                   data[3]))])
                    torch.save(pser[0], f"{filepath}/{data[0]}/protein_seq_repr.pth")
                if not os.path.exists(f"{filepath}/{data[0]}/smiles_repr.pth"):
                    sr = generate_smiles_reprs([extract_smiles_from_sdf(data[2])])
                    torch.save(sr.squeeze(dim=0), f"{filepath}/{data[0]}/smiles_repr.pth")
            except Exception as e:
                fileio.write_file_text(log_filepath, f"{data}: {e}\n", True)
                abnormal_samples += 1
        if abnormal_samples > 0:
            print(f"存在{len(abnormal_samples)}条错误样本, 信息已保存至{log_filepath}.")

    # 样本组装, 预特征读取
    def pre_assemble(self, pdb_ids: list[str], filepath: str) -> bool:
        ls_struct = []
        ls_seq = []
        ls_smiles = []
        try:
            for pdb_id in pdb_ids:
                pstr = torch.load(f"{filepath}/{pdb_id}/protein_struct_repr.pth", weights_only=False)
                pser = torch.load(f"{filepath}/{pdb_id}/protein_seq_repr.pth", weights_only=False)
                sr = torch.load(f"{filepath}/{pdb_id}/smiles_repr.pth", weights_only=False)
                ls_struct.append(pstr)
                ls_seq.append(pser)
                ls_smiles.append(sr)
            self.features["vs_struct"] = torch.stack(ls_struct)
            self.features["vs_seq"] = torch.stack(ls_seq)
            self.features["vs_smiles"] = torch.stack(ls_smiles)
        except Exception as e:
            print(f"组装过程发生错误: {e}.")


# EX.特征工程方法补充
# 1.蛋白质序列特征筛选
def filter_sequences_by_chains(seqs: dict[str], chains: str | None = None):
    if chains is not None:
        return "X".join([seqs.get(chain) for chain in chains])
    return "X".join(seqs.values())


