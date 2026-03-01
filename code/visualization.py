"""
visualization.py进行了项目使用过程中所用到的可视化部分:
"""
import math
import os
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_pipeline import parse_logfile, parse_logfile2
from aptheta_utils import fileio
from parameter import SEED, set_seed
from torch.utils.data import random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False

# 1.数据分布展示
# 思路: x轴表示绝对值, y轴表示百分比
def show_data_distribution(data, dataset_indices):
    x = np.array([math.log10(d[1]-d[2]+1) for d in data])
    y = np.array([(d[1]-d[2])/d[1]*100 for d in data])
    names = [(d[0], d[1], d[2]) for d in data]

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    category_colors = {1: "red", 2: "blue", 3: "green"}
    category_labels = {1: "训练集", 2: "验证集", 3: "测试集"}
    point_colors = [category_colors[di] for di in dataset_indices]
    # 创建单个scatter对象
    scatter = ax.scatter(x, y,
                        alpha=0.7, s=100,
                        c=point_colors,
                        edgecolors='w')
    
    legend_elements = [Patch(facecolor=color, label=category_labels[cat]) 
                    for cat, color in category_colors.items()]
    ax.legend(handles=legend_elements)
    # for category in [1, 2, 3]:
    #     mask = dataset_indices == category
    #     ax.scatter(x[mask], y[mask],
    #             alpha=0.7, s=100,
    #             c=category_colors[category],
    #             edgecolors='w',
    #             label=f"{category_name[category]}")

    # 设置坐标系
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(min(x), max(x)+1)
    ax.set_ylim(min(y), max(y)+10)
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xlabel("优化值(log10, 基础1偏移)", fontsize=12)
    ax.set_ylabel("优化百分比(%)", fontsize=12)
    ax.set_title("数据集分布", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(handles=legend_elements)

    # 添加注释
    annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def onclick(event):
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                idx = ind['ind'][0]
                annot.xy = (x[idx], y[idx])
                annot.set_text(f"{names[idx]}\n({x[idx]:.2f}, {y[idx]:.2f})")
                annot.set_text(f"pdb_id: {names[idx][0]}\n" +
                               f"最佳指标b: {names[idx][1]}\n" +
                               f"最佳指标a: {names[idx][2]}\n" +
                               f"优化值(log10(x+0.01)): {x[idx]:.2f}\n" +
                               f"优化百分比: {y[idx]:.2f}%")
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                annot.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.tight_layout()
    plt.show()


# 1.ex.数据分布展示直方图
# 思路: x轴表示绝对值, y轴表示百分比
def show_data_distribution_ex(data):
    data = np.array([(d[1]-d[2])/d[1]*100 for d in data])

    # 创建图形
    # plt.figure(figsize=(12, 8))

    bins = [-1e+7, 0, 5, 20, 35, 50, 65, 80, 95, 100]

    # plt.hist(data,
    #          bins=bins,
    #          edgecolor='black', 
    #          alpha=0.7, 
    #          color='skyblue',
    #          rwidth=0.9)

    # 设置刻度在每个区间的中心
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    labels = [
        "<0",
        "0-5",
        "5-20",
        "20-35",
        "35-50",
        "50-65",
        "65-80",
        "80-95",
        "95-100"
    ]

    cat = np.digitize(data, bins)

    # 统计
    counts = [np.sum(cat == i) for i in range(1, len(bins))]
    fig, ax = plt.subplots()

    bars = ax.bar(labels, counts)

    # 在柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,   # x 位置：柱子中间
            height,                              # y 位置：柱顶
            f'{int(height)}',                    # 显示的文本
            ha='center',                         # 水平居中
            va='bottom'                          # 文字在柱顶上方
    )
    # 画柱状图（等宽）
    plt.xticks(rotation=45)

    # plt.xticks(bin_centers, labels, rotation=45)

    # 设置图表标题和标签
    plt.title('数据分布直方图' , fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('指标变化范围（百分制）', fontsize=12)
    plt.ylabel('频数', fontsize=12)

    # 添加网格线
    # plt.grid(axis='y', alpha=0.3, linestyle='--')

    # # 在每个柱子上方显示频数
    # counts, bins, patches = plt.hist(data, bins=bins)
    # for i, (count, patch) in enumerate(zip(counts, patches)):
    #     plt.text(patch.get_x() + patch.get_width()/2, count + 1, 
    #             f'{int(count)}', 
    #             ha='center', va='bottom', fontweight='bold')

    # 添加统计信息
    plt.text(0.8, 0.98, f'总数据量: {len(data)}\n平均值: {np.mean(data):.3f}\n标准差: {np.std(data):.3f}', 
            transform=plt.gca().transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()

# 1.ex.数据分布展示直方图
# 输入: [(xb, xa, iw), ...]
def show_data_distribution_ex2(data):
    print("三数值直方图")
    data = np.array([100*(d[0]-d[2])/(d[0]-d[1]+1e-6) for d in data])

    bins = [-1e+9, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1e+9]

    # 设置刻度在每个区间的中心
    labels = [
        "<0",
        "0-10",
        "10-20",
        "20-30",
        "30-40",
        "40-50",
        "50-60",
        "60-70",
        "70-80",
        "80-90",
        "90-100",
        ">100"
    ]

    cat = np.digitize(data, bins)

    # 统计
    counts = [np.sum(cat == i) for i in range(1, len(bins))]
    fig, ax = plt.subplots()

    bars = ax.bar(labels, counts)

    # 在柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,   # x 位置：柱子中间
            height,                              # y 位置：柱顶
            f'{int(height)}',                    # 显示的文本
            ha='center',                         # 水平居中
            va='bottom'                          # 文字在柱顶上方
    )
    # 画柱状图（等宽）
    plt.xticks(rotation=45)

    # 设置图表标题和标签
    plt.title('数据分布直方图' , fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('优化情况比较（百分制）', fontsize=12)
    plt.ylabel('频数', fontsize=12)

        # 添加统计信息
    # plt.text(0.8, 0.98, f'总数据量: {len(data)}\n平均值: {np.mean(data):.3f}\n标准差: {np.std(data):.3f}', 
    #         transform=plt.gca().transAxes, 
    #         verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()


# 2.PDBBind数据蛋白链情况条形图
from featurer import extract_sequences_from_pdb
def show_pdbbind_chain(data_path):
    pdb_ids = [pdb_id for pdb_id in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, pdb_id))]
    data = np.array([extract_sequences_from_pdb(f"{data_path}/{pdb_id}/{pdb_id}_atoms.pdb").count(":")
                     for pdb_id in pdb_ids])
    values = range(0, np.max(data)+1)
    counts = [np.sum(data == i) for i in values]
    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制直方图，每20为一组
    bars = plt.bar(values, counts, edgecolor='black', alpha=0.7, color='steelblue')
    plt.title(f'PDBBind数据蛋白链情况条形图 (n={len(data)})')
    plt.xlabel('蛋白质链数')
    plt.ylabel('频数')
    plt.xticks(range(1, np.max(data)+2))
    plt.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# ex.工作流
def workflow(logfile: str):
    # 0.初始准备
    set_seed()

    # 1.创建初始数据集
    raw_dataset = parse_logfile(logfile, "./data/self-built")
    visual_dataset = [[rd[0], rd[4], rd[5]] for rd in raw_dataset]
    print(f"数据集大小: {len(visual_dataset)}")
    dataset_indices = np.array(range(len(visual_dataset)))

    # 2.数据集划分
    train_size = int(0.8 * len(raw_dataset))
    val_size = int(0.2 * train_size)
    train_size -= val_size
    test_size = len(raw_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(raw_dataset, [train_size, val_size, test_size])
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}, 测试集大小: {test_size}")
    dataset_indices[train_dataset.indices] = 1
    dataset_indices[val_dataset.indices] = 2
    dataset_indices[test_dataset.indices] = 3
    print(dataset_indices)

    show_data_distribution(visual_dataset, dataset_indices)
    show_data_distribution_ex(visual_dataset)

# ex.1数据分布直方图
def workflow2(logfile: str):
    # 0.初始准备
    set_seed()

    # 1.创建初始数据集
    raw_dataset = parse_logfile2(logfile, "./data/PDBbind_v2020_PL")
    visual_dataset = [[rd[0], rd[4], rd[5]] for rd in raw_dataset]
    print(f"数据集大小: {len(visual_dataset)}")
    dataset_indices = np.array(range(len(visual_dataset)))

    show_data_distribution_ex(visual_dataset)

# ex.1.1给定损失函数修改前后的标签数据，绘制第一阶段变化情况（标签数据指标的好坏）的数据分布直方图
def workflow21(log_b: str, log_a: str):
    # 1.创建初始数据集
    before_dataset = parse_logfile2(log_b, "./data/PDBbind_v2020_PL")
    after_dataset = parse_logfile2(log_a, "./data/PDBbind_v2020_PL")
    visual_dataset = []
    for i, rd in enumerate(after_dataset):
        visual_dataset.append([rd[0], before_dataset[i][5], rd[5]])
    print(f"数据集大小: {len(visual_dataset)}")
    show_data_distribution_ex(visual_dataset)


# ex.2数据分布直方图
def workflow3(logfile: str):
    # 0.初始准备
    set_seed()

    log = fileio.read_file_lines(logfile)
    if log[-1] == "":
        log.pop()
    data = []
    for l in log:
        d = l.split(",")
        xb = float(d[1].split(":")[1].strip())
        xa = float(d[2].split(":")[1].strip())
        iw = float(d[3].split(":")[1].strip()[:-1])
        data.append([xb, xa, iw])

    print(f"数据集大小: {len(data)}")
    data1 = [[row[1], row[0], *row[2:]] for row in data]
    show_data_distribution_ex(data1)
    show_data_distribution_ex2(data)


def show_data_bygptmethods(data):
    # =============================
    # 替换为你的数据
    # =============================
    # 示例：假设它们都是 numpy array
    target = np.array(d[0] for d in data)
    baseline_pred = np.array(d[1] for d in data)
    model_pred = np.array(d[2] for d in data)

    # =============================
    # 基础计算
    # =============================
    baseline_residual = baseline_pred - target
    model_residual = model_pred - target

    rmse_baseline = np.sqrt(mean_squared_error(target, baseline_pred))
    rmse_model = np.sqrt(mean_squared_error(target, model_pred))

    mae_baseline = mean_absolute_error(target, baseline_pred)
    mae_model = mean_absolute_error(target, model_pred)

    improvement = np.abs(baseline_residual) - np.abs(model_residual)

    # =============================
    # 1️⃣ Target vs Baseline
    # =============================
    plt.figure()
    plt.scatter(target, baseline_pred, alpha=0.3)
    plt.plot([target.min(), target.max()],
            [target.min(), target.max()])
    plt.xlabel("Target")
    plt.ylabel("Baseline Prediction")
    plt.title(f"Baseline vs Target (RMSE={rmse_baseline:.3f})")
    plt.tight_layout()
    plt.show()

    # =============================
    # 2️⃣ Target vs Model
    # =============================
    plt.figure()
    plt.scatter(target, model_pred, alpha=0.3)
    plt.plot([target.min(), target.max()],
            [target.min(), target.max()])
    plt.xlabel("Target")
    plt.ylabel("Model Prediction")
    plt.title(f"Model vs Target (RMSE={rmse_model:.3f})")
    plt.tight_layout()
    plt.show()

    # =============================
    # 3️⃣ 残差分布对比
    # =============================
    plt.figure()
    plt.hist(baseline_residual, bins=50, alpha=0.5, label="Baseline Residual")
    plt.hist(model_residual, bins=50, alpha=0.5, label="Model Residual")
    plt.axvline(0)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =============================
    # 4️⃣ RMSE / MAE 柱状图
    # =============================
    plt.figure()
    metrics = ["RMSE", "MAE"]
    baseline_values = [rmse_baseline, mae_baseline]
    model_values = [rmse_model, mae_model]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, baseline_values, width, label="Baseline")
    plt.bar(x + width/2, model_values, width, label="Model")

    plt.xticks(x, metrics)
    plt.ylabel("Error")
    plt.title("Error Metric Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =============================
    # 5️⃣ 改进量散点图（进阶）
    # =============================
    plt.figure()
    plt.scatter(target, improvement, alpha=0.3)
    plt.axhline(0)
    plt.xlabel("Target")
    plt.ylabel("Absolute Error Improvement")
    plt.title("Improvement over Baseline")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    set_seed()
    workflow3("./test/0228single/single.txt")
    # workflow3("s5229l3pd256.txt")
    # workflow3("s5229l3p256_nl.txt")
    # workflow2("test.txt")
    # workflow21("test_b.txt", "test_a.txt")
    # workflow("./log/20251023010518/result.txt")
    # show_pdbbind_chain("F:\\program\\test\\data")
    # show_pdbbind_chain("F:\\PDBbind\\2020\\PDBbind_v2020_PL")