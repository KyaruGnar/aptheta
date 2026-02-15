"""
featurer.py文件通过原始数据和预训练模型生成模型的特征输入, 是aptheta模型的特征工程:
1.蛋白质结构特征(由ProteinMPNN模型生成)
2.蛋白质序列特征(由esm2模型生成)
3.小分子SMILES特征(由ChemBERTa模型生成)

Last update: 2026-02-08 by Junlin_409
version: 1.1.0 Linux系统上加速尝试
"""

import argparse
import os
import random
import shutil
import sys
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from aptheta_utils import fileio
from parameter import set_seed
from mlxparam import MlXParamTrainer
from data_pipeline import DataPipeline, PDBBindDataSet


# 常量区
CPU_REQ_NUM = 96      # CPU需求数量(取更小)
CPU_REQ_PCT = 0.75    # CPU需求占比(取更小)


# 参数区
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_file", type=str)
parser.add_argument("--data_path", default="", type=str)
parser.add_argument("--dataset_type", default="pdbbind", choices=["self_built", "pdbbind"], type=str)
parser.add_argument("--log_path", type=str)
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

# 输出重定向
def redirect_stdout_stderr(log_file):
    if sys.platform == "linux" and os.path.exists("/dev/shm"):
        os.makedirs("/dev/shm/aptheta/log", exist_ok=True)
        log_file = (f"/dev/shm/aptheta/log/{os.path.split(log_file)[1]}")
    log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(log_fd, 1)  # stdout
    os.dup2(log_fd, 2)  # stderr
    os.close(log_fd)

# 样本处理
def process(sample):
    try:
        redirect_stdout_stderr(f"{sample[-1]}_run")
        mpt = MlXParamTrainer()
        mpt.setup_model(weight_decay=0.0)
        mpt.record(*sample)
        return (1, sample[0])
    except Exception as e:
        print(e)
        return (0, sample[0])

# 并行学习
def run(samples):
    n_jobs = max(1, min(CPU_REQ_NUM, int(os.cpu_count()*CPU_REQ_PCT)))
    # joblib 在后台使用 concurrent.futures
    parallel = Parallel(n_jobs=n_jobs, return_as="generator")
    # 启动 joblib 并生成结果生成器（不会一次性提交全部）
    results_iter = parallel(delayed(process)(sample) for sample in samples)
    sucess_count = 0
    failure_id = []
    with tqdm(total=len(samples), desc="Total Progress") as pbar:
        for r in results_iter:
            pbar.update(1)
            if r[0] == 1:
                sucess_count += 1
            else:
                failure_id.append(r[1])
    return sucess_count, failure_id

# 测试脚本函数
def test_func(sample):
    time.sleep(random.random())
    return sample

# 测试运行函数
def test_run():
    print("正在运行测试模式.")
    n_jobs = max(1, min(CPU_REQ_NUM, int(os.cpu_count()*CPU_REQ_PCT)))
    print(f"启用cpu数量: {n_jobs}.")
    parallel = Parallel(n_jobs=n_jobs, return_as="generator_unordered")
    samples = list(range(1, 257))
    results = []
    for result in tqdm(
        parallel(delayed(test_func)(sample) for sample in samples),
        total=len(samples),
        desc="Total Progress"
    ):
        results.append(result)
    print(f"运行结果: {results}.")

# 主函数
if __name__ == "__main__":
    set_seed()
    # 0.测试区
    if args.test:
        test_run()
        exit(0)
    # 1.数据集加载
    if args.log_path is None or args.dataset_file is None:
        raise ValueError()
    os.makedirs(args.log_path, exist_ok=True)
    if args.dataset_type == "self_built":
        dataset = DataPipeline.load(args.dataset_file, args.data_path)
    elif args.dataset_type == "pdbbind":
        dataset = PDBBindDataSet.load(args.dataset_file, args.data_path)
    else:
        raise ValueError()
    # 2.样本产生
    samples_with_log = [dataset.prepare_sample(pdb_id, precise_pocket=True)
                        + (os.path.normpath(f"{args.log_path}/{pdb_id}"),)
                        for pdb_id in dataset.dataset]
    # 3.运行
    result = run(samples_with_log)
    print(f"任务总数: {len(samples_with_log)}, 成功任务数: {result[0]}. 失败样本将保存至文件.")
    fileio.write_file_lines(os.path.normpath(f"{args.log_path}/failure_ids.txt"), result[1])
    if sys.platform == "linux" and os.path.exists("/dev/shm"):
        shutil.move("/dev/shm/aptheta/log", f"{args.log_path}_run")
