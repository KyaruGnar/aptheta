"""
iweights.py建立了一个模型:
input: mlxparam.py产生的log文件信息(含蛋白质pdb文件, 小分子, 指标变化, 权重变化)
output: 权重变化预测矩阵

Last update: 2026-02-15 by Junlin_409
version: 1.0.0 标记
"""


# 导入区
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score # type: ignore
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader, Dataset
from data_pipeline import parse_logfile, parse_logfile2, DataPipeline, PDBBindDataSet
from parameter import SEED
from featurer import FeatureCollector
from aptheta_utils import fileio
from extcall import rmsd_eval, vina

# 零.初始准备
# 1.随机准备
def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 2.设备准备
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# DEVICE = torch.device("cpu")


# 二.数据集
class ProteinDataset(Dataset):
    def __init__(self, raw_dataset):
        self.iweights = torch.tensor([d[6] for d in raw_dataset])
    
    def pre_generate(self, pdb_ids, filepath):
        fc = FeatureCollector()
        fc.pre_assemble(pdb_ids, filepath)
        self.vs_struct = fc.features["vs_struct"]
        self.vs_seq = fc.features["vs_seq"]
        self.vs_smiles = fc.features["vs_smiles"]

    def generate(self, raw_dataset):
        fc = FeatureCollector()
        fc.generate(raw_dataset)
        fc.save("PDBbind2020_PL.pt")
        self.vs_struct = fc.features["vs_struct"]
        self.vs_seq = fc.features["vs_seq"]
        self.vs_smiles = fc.features["vs_smiles"]

    def __len__(self) -> int:
        return len(self.iweights)

    def __getitem__(self, idx: int):
        v_struct = self.vs_struct[idx]
        v_seq = self.vs_seq[idx]
        v_smile = self.vs_smiles[idx]
        iweight = self.iweights[idx]
        return v_struct, v_seq, v_smile, iweight


# 三.模型
class IWeight(nn.Module):
    def __init__(self, proj_dim=256, output_dim=6, dropout=0.1):
        super(IWeight, self).__init__()
        # 映射层
        self.proj_pro_struct = nn.Linear(128, proj_dim)
        self.proj_pro_seq = nn.Linear(1280, proj_dim)
        self.proj_smiles = nn.Linear(384, proj_dim)
        # 归一化层
        self.pf1_out = nn.LayerNorm(proj_dim)
        self.pf2_out = nn.LayerNorm(proj_dim)
        self.pf3_out = nn.LayerNorm(proj_dim)
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim*3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, v_struct, v_seq, v_smiles):
        pf1 = self.pf1_out(self.proj_pro_struct(v_struct))
        pf2 = self.pf2_out(self.proj_pro_seq(v_seq))
        pf3 = self.pf3_out(self.proj_smiles(v_smiles))
        pfc = torch.cat([pf1, pf2, pf3], dim=-1)
        return self.mlp(pfc)


# 四.训练器
class IWeightTrainer:
    def __init__(self, device: str = "auto"):
        self.device = self.setup_device(device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        # 训练历史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_rmse': [], 'val_rmse': [],
            'lr': []
        }

    def setup_device(self, device: str) -> torch.device:
        if device == 'auto':
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def setup_model(self, proj_dim: int = 256, output_dim: int = 6, dropout: float = 0.1,
                   learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        self.model = IWeight(proj_dim, output_dim, dropout).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           mode='min',
                                           factor=0.5,
                                           patience=10,
                                           verbose=True,
                                           min_lr=1e-6)

    def train_epoch(self, train_loader: DataLoader, loss_fn: nn.Module):
        # 初始化
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        # 每批次训练
        for vs_struct, vs_seq, vs_smiles, iweights in train_loader:
            # 输入准备
            vs_struct = vs_struct.to(self.device)
            vs_seq = vs_seq.to(self.device)
            vs_smiles = vs_smiles.to(self.device)
            iweights = iweights.to(self.device)
            # 输出与损失计算
            self.optimizer.zero_grad()
            outputs = self.model(vs_struct, vs_seq, vs_smiles)
            loss = loss_fn(outputs, iweights)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            # 结果收录
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(iweights.cpu().numpy())
        # 指标计算
        avg_loss = total_loss / max(len(train_loader), 1)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        # 返回
        return avg_loss, rmse

    def validate(self, val_loader: DataLoader, loss_fn: nn.Module):
        # 初始化
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        # 每批次验证
        with torch.no_grad():
            for vs_struct, vs_seq, vs_smiles, iweights in val_loader:
                # 输入准备
                vs_struct = vs_struct.to(self.device)
                vs_seq = vs_seq.to(self.device)
                vs_smiles = vs_smiles.to(self.device)
                iweights = iweights.to(self.device)
                # 输出与损失计算
                outputs = self.model(vs_struct, vs_seq, vs_smiles)
                loss = loss_fn(outputs, iweights)
                total_loss += loss.item()
                # 结果收录
                all_preds.extend(outputs.detach().cpu().numpy())
                all_targets.extend(iweights.cpu().numpy())
        # 指标计算
        avg_loss = total_loss / max(len(val_loader), 1)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        # 返回
        return avg_loss, rmse

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, patience: int = 20, loss_fn: nn.Module = None):
        # 初始设置
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        # 训练可视化输出
        print(f"开始训练，设备: {self.device}")
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Train R2':<10} {'Val R2':<10} {'LR':<10}")
        print("-" * 70)
        # 每epoch训练
        for epoch in range(epochs):
            # 训练
            train_loss, train_rmse = self.train_epoch(train_loader, loss_fn)
            # 验证
            val_loss, val_rmse = self.validate(val_loader, loss_fn)
            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            # 保存历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            self.history['lr'].append(current_lr)
            # 打印进度
            print(f"{epoch+1:>4}/{epochs} {train_loss:>10.6f} {val_loss:>10.6f} {current_lr:>9.2e}")
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n早停触发！在epoch {epoch+1}停止训练")
                    break
        print("\n训练完成！")

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for vs_struct, vs_seq, vs_smiles, _ in test_loader:
                vs_struct = vs_struct.to(self.device)
                vs_seq = vs_seq.to(self.device)
                vs_smiles = vs_smiles.to(self.device)
                outputs = self.model(vs_struct, vs_seq, vs_smiles)
                all_preds.extend(outputs.cpu().numpy())
        return all_preds

    def plot_training_history(self):
        fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 10))
        # 损失曲线
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        # RMSE曲线
        ax2.plot(self.history['train_rmse'], label='Train RMSE')
        ax2.plot(self.history['val_rmse'], label='Val RMSE')
        ax2.set_title('Training and Validation RMSE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE')
        ax2.legend()
        ax2.grid(True)
        # 学习率曲线
        ax3.plot(self.history['val_loss'], label='Val Loss', color='red')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation Loss', color='red')
        ax3.tick_params(axis='y', labelcolor='red')
        ax3_2 = ax3.twinx()
        ax3_2.plot(self.history['lr'], label='Learning Rate', color='blue', linestyle='--')
        ax3_2.set_ylabel('Learning Rate', color='blue')
        ax3_2.tick_params(axis='y', labelcolor='blue')
        ax3_2.set_title('Learning Rate Schedule')
        # 加载并保存
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


# 五.IWeight模型工作流
def workflow(logfile: str, seed: int = SEED):
    # 0.初始准备
    set_seed(seed)

    # 1.创建初始数据集
    raw_dataset = parse_logfile(logfile, "./data/self-built")
    print(f"数据集大小: {len(raw_dataset)}")
    dataset = ProteinDataset(raw_dataset)
    dataset.generate(raw_dataset)

    # 2.数据集划分
    train_size = int(0.8 * len(dataset))
    val_size = int(0.2 * train_size)
    train_size -= val_size
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}, 测试集大小: {test_size}")

    # 3.创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 4.初始化训练器
    trainer = IWeightTrainer()
    trainer.setup_model()

    # 5. 开始训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=300,
        patience=45,
        loss_fn=nn.MSELoss()
    )

    # 6. 绘制训练历史
    trainer.plot_training_history()

    # 7.加载最佳模型并进行预测
    checkpoint = torch.load('best_model.pth')
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    predictions = trainer.predict(test_loader)
    print(f"预测结果: {predictions}")

    dp = DataPipeline()
    for i, di in enumerate(test_dataset.indices):
        rd = raw_dataset[di]
        sp, ifs, b = dp.prepare_sample(rd[0], precise_pocket=True)
        weights = {
            "Gauss1": -0.035579 * (1+predictions[i][0]),
            "Gauss2": -0.005156 * (1+predictions[i][1]),
            "Repulsion": 0.840245 * (1+predictions[i][2]),
            "Hydrophobic": -0.035069 * (1+predictions[i][3]),
            "Hydrogen bonding": -0.587439 * (1+predictions[i][4]),
            "Glue": 50,
            "Rot": 0.05846 * (1+predictions[i][5])
        }
        ofp = f"{sp}/out_weights.pdbqt"
        vina(ifs, b, ofp, weights)
        rmsd = rmsd_eval(ifs, ofp)
        res = f"pdb_id: {rd[0]}, xb_rmsd: {rd[4]}, xa_rmsd: {rd[5]}, iw_rmsd: {rmsd[0]}.\n"
        print(res)
        fileio.write_file_text("pred_res.txt", res, append=True)

# 五.IWeight模型工作流(基于预训练预特征的PDBBIND数据集)
def workflow_bypre(name: str, pdb_ids: list[str], test_ids: list[str], log_filepath: str, feature_filepath: str, seed: int = SEED):
    # 0.初始准备
    set_seed(seed)

    # 1.创建初始数据集
    logfile = "log.txt"
    if os.path.exists(logfile):
        os.remove(logfile)
    for pdb_id in pdb_ids:
        plog = fileio.read_file_lines(f"{log_filepath}/{pdb_id}")
        if plog[-1] == "":
            plog.pop()
        fileio.write_file_lines(logfile, plog, True)
    raw_dataset = parse_logfile2(logfile, "./data/PDBbind_v2020_PL")
    print(f"数据集大小: {len(raw_dataset)}")
    dataset = ProteinDataset(raw_dataset)
    dataset.pre_generate(pdb_ids, feature_filepath)
    # 1.2.创建测试集
    test_logfile = "test.txt"
    if not os.path.exists(test_logfile):
        for pdb_id in test_ids:
            plog = fileio.read_file_lines(f"{log_filepath}/{pdb_id}")
            if plog[-1] == "":
                plog.pop()
            fileio.write_file_lines(test_logfile, plog, True)
    test_samples = parse_logfile2(test_logfile, "./data/PDBbind_v2020_PL")
    test_dataset = ProteinDataset(test_samples)
    test_dataset.pre_generate(test_ids, feature_filepath)

    # 2.数据集划分
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")

    # 3.创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 4.初始化训练器
    trainer = IWeightTrainer()
    trainer.setup_model()

    # 5. 开始训练
    # trainer.train(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     epochs=300,
    #     patience=45,
    #     loss_fn=nn.MSELoss()
    # )

    # # # 6. 绘制训练历史
    # trainer.plot_training_history()

    # 7.加载最佳模型并进行预测
    checkpoint = torch.load('best_model.pth')
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    predictions = trainer.predict(test_loader)
    print(f"预测结果: {predictions}")

    # dp = PDBBindDataSet("./data/PDBbind_v2020_PL")
    # for i, t in enumerate(test_dataset):
    #     rd = test_samples[i]
    #     sp, ifs, b = dp.prepare_sample(rd[0], precise_pocket=True)
    #     print(f"pdb_id: {rd[0]}")
    #     weights = {
    #         "Gauss1": -0.035579 * (1+predictions[i][0]),
    #         "Gauss2": -0.005156 * (1+predictions[i][1]),
    #         "Repulsion": 0.840245 * (1+predictions[i][2]),
    #         "Hydrophobic": -0.035069 * (1+predictions[i][3]),
    #         "Hydrogen bonding": -0.587439 * (1+predictions[i][4]),
    #         "Glue": 50,
    #         "Rot": 0.05846 * (1+predictions[i][5])
    #     }
    #     ofp = f"{sp}/out_weights.pdbqt"
    #     vina(ifs, b, ofp, weights)
    #     rmsd = rmsd_eval(ifs, ofp)
    #     res = f"pdb_id: {rd[0]}, xb_rmsd: {rd[4]}, xa_rmsd: {rd[5]}, iw_rmsd: {rmsd[0]}.\n"
    #     print(res)
    #     if rd[0] != "5lch":
    #         os.remove(ofp)
    #     fileio.write_file_text(f"{name}.txt", res, append=True)


if __name__ == "__main__":
    # workflow("./log/20251023010518/result.txt")
    pass
