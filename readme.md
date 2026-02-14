
项目代码结构：
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
│   ├── repro_vina/             # 复现Vina评分的包
│   ├── aptheta_utils/          # 工具包
│   ├── extcall.py              # 外部调用归纳
│   ├── data_pipeline.py        # 数据工程
│   ├── featurer.py             # 特征工程
│   ├── mlxparam.py             # 标签工程
│   
└── test/                       # 测试
