
##项目代码结构：

aptheta/

├── data/                       # 数据

│   ├── self-bulit/             # 自建数据集

│   │   ├── pdb_ids.txt         # 蛋白质id集合

│   │   └── {pdb_id}/           # 样本数据

│   │       ├── protein/        # 蛋白质数据*

│   │       ├── ligands/        # 配体数据*

│   │       ├── mlxparam/       # 优化对接数据*

│   │       └── {pdb_id}.json   # 蛋白质信息

│   └── PDBbind_v2020_PL        # PDBBind2020数据集中的蛋白质-配体部分

│       ├── dataset.txt         # (成功预处理的)蛋白质id集合

│       └── {pdb_id}/           # 样本数据

│           ├── {pdb_id}_protein.xxx            # 蛋白质数据

│           ├── {pdb_id}_atoms.pdb              # 蛋白质原子坐标

│           ├── {pdb_id}_ligand.xxx             # 配体数据

│           ├── {pdb_id}_ligand_adapted.sdf     # 适应配体数据

│           └── mlxparam/                       # 优化对接数据

├── external/                   # 外部程序/模型

│   ├── AutoDock/               # AutoDock的受/配体格式转换脚本

│   ├── Vina/                   # Vina对接脚本

│   ├── DockRMSD_YF/            # DockRMSD指标计算脚本

│   ├── ChemBERTa-77M-MTR/      # ChemBERTa-77M-MTR模型

│   ├── esm2_t33_650M_UR50D.pt  # esm2模型

|

├── code/                       # 代码

│   ├── repro_vina/             # 复现Vina评分的包 (还提供了参考样例)

│           ├── simulation.py   # 结构的仿真模型

│           ├── docking.py      # 对接的实现模型

│           ├── parser.py       # pdbqt文件解析的集合

│           └── utility.py      # 其他工具类

│   ├── aptheta_utils/          # 工具包

│           ├── fileio.py       # 文件读取

│           └── timer.py        # 时间操作的包装

│   ├── parameter.py            # 部分公共变量存取, 设置种子和机器

│   ├── extcall.py              # 外部调用归纳

│   ├── data_pipeline.py        # 数据工程

│   ├── featurer.py             # 特征工程

│   ├── mlxparam.py             # 标签工程 (产生自适应权重)

│   ├── parallel.py             # 标签工程的并行计算脚本 (可通过命令行调用)

│   ├── iweight.py              # 预测权重的学习模型 (含工作流)

│   ├── fix.py                  # 旧版本代码运行结果的修正以及可能的数据筛选

│   ├── dock_score.py           # 对接分数实现 (复现包的部分功能截取)

│   ├── efficient_model.py      # 对接分数部分计算的高速化

│   ├── aptheta.py              # 命令行调用脚本合集

│   

└── env/                        # 环境

│   ├── install_linux.sh        # linux安装环境脚本

│   └── install_windows.ps1     # windows安装环境脚本



##命令行调用
通过aptheta.py提供了部分命令行调用的功能。
准备：虚拟环境安装完成并激活，在program根目录下执行：
python ./code/aptheta.py 子命令 参数
以下为示例，无法直接调用，需要根据需求自行调整，具体可使用--help查看子命令或者子命令下的参数
1.给定pdb_ids的自建数据集构建
python ./code/aptheta.py build_self --dataset dataset.txt --py2 D:/soft/MGLTools-1.5.6/python
2.PDBBind数据集构建
python ./code/aptheta.py build_pdbbind --dataset ./data/PDBbind_v2020_PL --py2 D:/soft/MGLTools-1.5.6/python
3.自建数据集的自适应权重产生
python ./code/aptheta.py mlxparam --dataset_file dataset.txt --data_path ./data/self_built ----logfile log_s.txt
4.自建数据集的预测权重训练
python ./code/aptheta.py iweight --logfile log_s.txt
5.PDBBind数据集的预特征产生
python ./code/aptheta.py prefeat --dataset_file dataset.txt --data_path ./data/PDBbind_v2020_PL --feature_path ./features
6.PDBBind数据集的预测权重训练
python ./code/aptheta.py iweight_bypre --name "1234" --prelogs ./log_p --prefeats ./features
7.PDBBind数据集的自适应权重产生*(该部分独立在parallel.py中)
python ./code/parallel.py --dataset_file filter_chain1.txt --data_path ./data/PDBbind_v2020_PL --log_path log_p