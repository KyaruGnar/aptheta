"""
mlxparam.py文件通过原始数据和机器学习生成模型的标签输入(优化权重), 是aptheta模型的标签工程:
1.优化权重的机器学习模型
2.损失函数
3.优化权重训练器
input: 原始蛋白质和小分子数据
output: 含对接更优权重的log信息

Last update: 2026-02-15 by Junlin_409
version: 1.2.0 更新weights
"""

# 引入区
import copy
import os
import shutil
import sys
import torch
from torch import nn, optim
from extcall import vina, rmsd_eval
from aptheta_utils.fileio import write_file_lines
from dock_score import DockingModelWithPoses
from parameter import DEVICE, VINA_WEIGHTS
from repro_vina.parser import parse_receptor_pdbqts, parse_ligand_pdbqts, parse_output_pdbqt_simply


# 一.权重的机器学习模型(通过机器学习迭代出一个指标较好的权重作为后续aptheta模型训练的target)
class MlXParam(nn.Module):
    def __init__(self, device: torch.device = DEVICE) -> None:
        super().__init__()
        self.weights = nn.ParameterDict()
        self.model: DockingModelWithPoses | None = None
        self.device = device
        self.initialize_weights()

    def initialize_weights(self) -> None:
        if len(self.weights) == 0:
            for name, weight in VINA_WEIGHTS.items():
                self.weights[name] = nn.Parameter(torch.tensor(weight, dtype=torch.float64, device=self.device))
            return
        with torch.no_grad():
            for name, weight in VINA_WEIGHTS.items():
                self.weights[name].copy_(torch.tensor(weight, dtype=torch.float64, device=self.device))

    def initialize_model(self, receptor: str, ligands: list[str]) -> None:
        self.model = DockingModelWithPoses(self.weights) # type: ignore
        self.model.set_receptor(*parse_receptor_pdbqts(receptor))
        self.model.set_ligands(parse_ligand_pdbqts(ligands))
        for atom in self.model.receptor.valid_atoms:
            atom.coordinate = torch.tensor(atom.coordinate, dtype=torch.float64, device=self.device) # type: ignore

    def forward(self, output_file: str):
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        self.model.set_poses(parse_output_pdbqt_simply(output_file))
        for atom, poses in self.model.poses.items():
            self.model.poses[atom] = [torch.tensor(p, dtype=torch.float64, device=self.device) # type: ignore
                                      for p in poses] # type: ignore
        return self.model.vina_with_poses(print_mode=False, ml_mode=True)

# 二.损失函数区
# 1.vina对接分数损失
def vina_loss(energies, print_mode: bool = False):
    vina_min = energies[1][0]
    for i, e in enumerate(energies):
        if i == 0:
            continue
        vina_min = torch.min(vina_min, e[0])
    loss = energies[0][0] - vina_min
    if print_mode:
        print(f"Target: {energies[0][0]:.5f}; Min_score: {vina_min:.5f}; Vina_loss: {loss:.5f}")
    return loss

# 2.权重学习损失
def weights_loss(inital_weights: nn.ParameterDict, weights: nn.ParameterDict, print_mode: bool = False):
    loss = 0.0
    for key in weights:
        loss += torch.tan(
            torch.pow(
                torch.log(weights[key] / inital_weights[key])
                /torch.log(torch.tensor(4)), 2)
            * torch.pi / 2)
        if print_mode:
            print(f"{key}: {weights[key].item():.7f}", end="; ")
    if print_mode:
        print(f"Weights_loss: {loss:.5f}")
    return loss

# 三.优化权重训练器
class MlXParamTrainer:
    def __init__(self, inner_epoch: int = 100, outer_epoch: int = 30, tolerate: int = 6) -> None:
        self.model: MlXParam | None = None
        self.optimizer: optim.Optimizer | None = None
        self.inner_epoch = inner_epoch
        self.outer_epoch = outer_epoch
        self.tolerate = tolerate

    def setup_model(self, learning_rate: float = 1e-4, weight_decay: float = 1e-5,
                    device: torch.device = DEVICE) -> None:
        self.model = MlXParam(device).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 内循环参数学习
    def inner_train(self, output_file: str, print_mode: bool = False) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not initialized.")
        self.model.train()
        inital_weights = copy.deepcopy(self.model.weights)
        best_state_dict = None
        best_loss = 1e9+7
        best_epoch = -1
        for epoch in range(self.inner_epoch):
            if print_mode:
                print(f"Inner epoch {epoch+1:03}.", flush=True)
            self.optimizer.zero_grad()
            results = self.model(output_file)
            loss = vina_loss(results, print_mode) + weights_loss(inital_weights, self.model.weights, print_mode)
            loss.backward()
            self.optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state_dict = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
        if print_mode:
            print(f"best_epoch: {best_epoch+1}; best_loss:{best_loss:.5f}")
        self.model.load_state_dict(best_state_dict) # type: ignore

    # 外循环参数学习
    def outer_train(self, sample_path: str, input_files: dict, box: dict,
                    print_mode: bool = False) -> tuple[nn.ParameterDict, list[float], int]:
        """
        EX2.数据存储原则(默认):\n
        (3)对接文件目录: DATA_PATH/{pdb_id}/mlxparam;
        """
        if sys.platform != "linux" or not os.path.exists("/dev/shm"):
            speed_up = False
        else:
            speed_up = True
            os.makedirs("/dev/shm/aptheta", exist_ok=True)
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        self.model.initialize_weights()
        self.model.initialize_model(input_files["receptor"], input_files["ligands"])
        output_file = f"{sample_path}/mlxparam/out0.pdbqt"
        if speed_up:
            output_file = f"/dev/shm/aptheta/mlxparam/{os.path.split(sample_path)[1]}/out0.pdbqt"
        vina(input_files, box, output_file, {k: v.detach().cpu().item() for k, v in self.model.weights.items()})
        best_index = rmsd_eval(input_files, output_file)
        if len(best_index) == 0:
            raise RuntimeError("指标评估发生错误.")
        best_weights = copy.deepcopy(self.model.weights)
        best_epoch = -1
        for epoch in range(self.outer_epoch):
            if print_mode:
                print(f"Outer epoch {epoch+1:03}, based on '{output_file}'.")
            self.inner_train(output_file, print_mode)
            new_output_file = f"{sample_path}/mlxparam/out{epoch+1}.pdbqt"
            if speed_up:
                new_output_file = f"/dev/shm/aptheta/mlxparam/{os.path.split(sample_path)[1]}/out{epoch+1}.pdbqt"
            vina(input_files, box, new_output_file, {k: v.detach().cpu().item() for k, v in self.model.weights.items()})
            new_index = rmsd_eval(input_files, new_output_file)
            if len(new_index) == 0:
                raise RuntimeError("指标评估发生错误.")
            if print_mode:
                print(f"Old rmsds: {best_index}; new rmsds: {new_index}")
            if best_index[0] > new_index[0]:
                best_weights = copy.deepcopy(self.model.weights)
                best_index = new_index
                best_epoch = epoch
                output_file = new_output_file
                if print_mode:
                    print(f"Update output in outer epoch {epoch+1:03}.")
            else:
                if epoch - best_epoch == self.tolerate:
                    if print_mode:
                        print(f"Reset weights in outer epoch {epoch+1:03}.")
                    self.model.initialize_weights()
                elif epoch - best_epoch  == 2 * self.tolerate:
                    if print_mode:
                        print(f"Break out in outer epoch {epoch+1:03}.")
                    break
                if print_mode:
                    print(f"Retain output of outer epoch {best_epoch+1:03}.")
            if speed_up:
                shutil.move(f"/dev/shm/aptheta/mlxparam/{os.path.split(sample_path)[1]}", f"{sample_path}/mlxparam")
        return best_weights, best_index, best_epoch

    def record(self, sample_path: str, input_files: dict, box: dict, filepath: str, print_mode: bool = False):
        best_weights, best_index, best_epoch = self.outer_train(sample_path, input_files, box, print_mode)
        output_file = f"{sample_path}/mlxparam/out0.pdbqt"
        index_0 = rmsd_eval(input_files, output_file)
        ws = [
            "Receptor",
            f"{input_files["receptor"]}",
            "ligand",
            f"{input_files["ligands"]}",
            "Original rmsds",
            f"{index_0}",
            "Best rmsds",
            f"{best_index}",
            "Best weights"
        ]
        for k, v in best_weights.items():
            ws.append(f"{k}: {v.item():.7f}")
        write_file_lines(filepath, ws, True)
        if print_mode:
            print(f"Original rmsds are {index_0}, the best rmsds are {best_index}, the best weights are", flush=True)
            for k, v in best_weights.items():
                print(f"{k}: {v.item():.7f}", end=" ")
            print()
