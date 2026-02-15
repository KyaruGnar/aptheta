"""
aptheta.py文件提供了命令行功能:


Last update: 2026-02-15 by Junlin_hpc
version: 1.0.0
"""

# 导入区
import click
import os
from aptheta_utils import fileio
from data_pipeline import DataPipeline, PDBBindDataSet
from mlxparam import MlXParamTrainer
from iweight import workflow, workflow_bypre
from featurer import FeatureCollector
from parameter import set_seed

# ==============================
# 主入口（Group）
# ==============================
@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """
    Aptheta Toolkit
    """
    # 如果没有输入子命令
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ==============================
# 子命令 1
# ==============================
@cli.command()
@click.option(
    "--dataset",
    type=click.Path(exists=True),
    help="将要构建的pdb_ids数据集的文件路径(一行一个pdb_id), 否则根据设定大小随机构建"
)
@click.option(
    "--data_size",
    type=int,
    help="设定随机构建时的数据集大小"
)
def build_self(dataset, data_size):
    """
    Prepare dataset from RCSB.
    """
    if not dataset and not data_size:
        raise click.UsageError("You must provide either --pdb_ids or --data_size.")
    if dataset:
        pdb_ids = fileio.read_file_lines(dataset)
        if pdb_ids[-1] == "":
            pdb_ids.pop()
    if not data_size:
        data_size = len(pdb_ids)
    dp = DataPipeline(data_size)
    dp.run(pdb_ids)


# ==============================
# 子命令 2
# ==============================
@cli.command()
@click.option(
    "--dataset",
    required=True,
    type=click.Path(exists=True),
    help="将要构建的PDBBind数据集的文件路径"
)
def build_pdbbind(dataset):
    """
    Prepare dataset from PDBBind.
    """
    dp = PDBBindDataSet(dataset)
    dp.run()

# ==============================
# 子命令 3
# ==============================
@cli.command()
@click.option(
    "--dataset_file",
    required=True,
    type=click.Path(exists=True),
    help="数据集文件路径"
)
@click.option(
    "--data_path",
    required=True,
    type=click.Path(exists=True),
    help="数据路径"
)
@click.option(
    "--logfile",
    required=True,
    type=click.Path(exists=True),
    help="优化权重信息存放文件路径"
)
def mlxparam(dataset_file, data_path, logfile):
    """
    常规方式生成自建数据集的优化权重
    """
    dp = DataPipeline.load(dataset_file, data_path)
    mlxp = MlXParamTrainer()
    mlxp.setup_model(weight_decay=0.0)
    for d in dp.dataset:
        try:
            sp, ifs, b = dp.prepare_sample(d, precise_pocket=True)
            mlxp.record(sp, ifs, b, logfile)
        except Exception as e:
            print(f"pdb {d}训练过程中出错，此样本作废.")


# ==============================
# 子命令 4
# ==============================
@cli.command()
@click.option(
    "--logfile",
    required=True,
    type=click.Path(exists=True),
    help="优化权重信息存放文件路径"
)
def iweight(logfile):
    """
    常规方式的自建数据集的预测优化权重工作流
    """
    workflow(logfile)


# ==============================
# 子命令 5
# ==============================
@cli.command()
@click.option(
    "--dataset_file",
    required=True,
    type=click.Path(exists=True),
    help="数据集文件路径"
)
@click.option(
    "--data_path",
    required=True,
    type=click.Path(exists=True),
    help="数据路径"
)
@click.option(
    "--logfile",
    required=True,
    type=click.Path(exists=True),
    help="预特征存放路径"
)
def prefeat(dataset_file, data_path, feature_path):
    """
    基于PDBBind数据集的预先特征训练
    """
    dataset = fileio.read_file_lines(dataset_file)
    if dataset[-1] == "":
        dataset.pop()
    raw_dataset = []
    for d in dataset:
        raw_dataset.append([d, f"{data_path}/{d}/{d}_protein_atoms.pdb", f"{data_path}/{d}/{d}_ligand_adapted.sdf", None])
    fc = FeatureCollector()
    fc.pre_generate(raw_dataset, feature_path)


# ==============================
# 子命令 6
# ==============================
@cli.command()
@click.option(
    "--name",
    default="predictions.txt",
    type=click.Path(exists=True),
    help="预测结果文件名"
)
@click.option(
    "--prelogs",
    required=True,
    type=click.Path(exists=True),
    help="优化权重信息文件路径"
)
@click.option(
    "--prefeats",
    required=True,
    type=click.Path(exists=True),
    help="预特征文件路径"
)
@click.option(
    "--prefeats",
    required=True,
    type=click.Path(exists=True),
    help="预特征存放路径"
)
@click.option(
    "--train_nums",
    type=int,
    help="训练集数量"
)
@click.option(
    "--test_nums",
    type=int,
    help="测试集数量(默认从倒数开始取)"
)
def iweight_bypre(name, prelogs, prefeats, train_nums, test_nums):
    """
    基于PDBBind数据集和预先特征的工作流
    """
    pdb_ids = []
    for pl in os.listdir(prelogs):
        if pl == "failure_ids.txt":
            continue
        pdb_ids.append(pl)
    if not train_nums or train_nums > len(pdb_ids):
        train_nums = len(pdb_ids)
    workflow_bypre(name, pdb_ids[:train_nums], pdb_ids[-test_nums:], prelogs, prefeats)


if __name__ == "__main__":
    set_seed()
    cli()
