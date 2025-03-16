import json
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D  # 用于创建自定义图例句柄

plt.rcParams['font.family'] = 'DejaVu Sans'  # 所有系统都包含的Unicode字体
plt.rcParams['axes.unicode_minus'] = False   # 禁用unicode负号

# --------------------------
# 配置参数
# --------------------------
DATA_DIR = "C:/nnUNet_v2_project/nnUNet/SelfCode_val/others/data_coculate/json_files"  # JSON文件存放路径
MODELS = ["ModelA", "ModelB"]  # 待比较的两个模型名称
ENSEMBLE_SUFFIX = "_ensemble"  # 集成文件标识符

# 颜色配置
FOLD_COLORMAP = LinearSegmentedColormap.from_list("fold", ["#2b8cbe", "#a6bddb"])  # 子模型颜色梯度
ENSEMBLE_COLORS = ["#e41a1c", "#377eb8"]  # 为每个模型的集成指定不同颜色

# --------------------------
# 数据加载（已适配.json数据结构）
# --------------------------
def parse_fold_number(filename):
    """使用正则表达式精确提取折号（适配类似 ModelA_fold0.json 或 ModelA_fold_1.json 的格式）"""
    match = re.search(r"fold[\W_]*(\d+)", filename)
    return int(match.group(1)) if match else None


def load_model_data(model_name):
    """加载指定模型的所有五折和集成数据（完全适配您的JSON结构）"""
    data = {"folds": [], "ensemble": None}

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".json") or model_name not in fname:
            continue

        filepath = os.path.join(DATA_DIR, fname)
        with open(filepath) as f:
            json_data = json.load(f)

        # 关键数据路径：使用 foreground_mean 下的 Dice（与您的数据结构完全匹配）
        dice = json_data["foreground_mean"]["Dice"]

        # 处理五折文件
        if "fold" in fname.lower():
            fold = parse_fold_number(fname)
            if fold is not None and 0 <= fold < 5:  # 确保是有效的五折文件
                data["folds"].append((fold, dice))

        # 处理集成文件
        elif ENSEMBLE_SUFFIX in fname.lower():
            data["ensemble"] = dice

    # 按折号排序并验证数据完整性
    data["folds"].sort(key=lambda x: x[0])
    if len(data["folds"]) != 5 or data["ensemble"] is None:
        raise ValueError(
            f"{model_name}数据不完整，检测到{len(data['folds'])}折和{'有' if data['ensemble'] else '无'}集成数据")

    return data


# --------------------------
# 可视化逻辑
# --------------------------
def plot_model_comparison(model_data, model_name):
    plt.figure(figsize=(10, 6))

    # 提取数据
    folds = [item[1] for item in model_data["folds"]]
    ensemble_dice = model_data["ensemble"]

    # ----------------------
    # 子模型性能散点图
    # ----------------------
    # 生成颜色梯度
    colors = FOLD_COLORMAP(np.linspace(0, 1, len(folds)))

    # 绘制每个折的结果
    for i, dice in enumerate(folds):
        fold_num = i
        plt.scatter(i, dice, color=colors[i], s=100,
                    edgecolor='white', zorder=3,
                    label=f'Fold {fold_num + 1}')

    # 添加子模型均值和标准差
    mean = np.mean(folds)
    std = np.std(folds)
    plt.axhspan(mean - std, mean + std, alpha=0.2, color='grey',
                label='5-Fold Mean±Std')
    plt.axhline(mean, color='grey', linestyle='--', linewidth=1)

    # ----------------------
    # 集成模型对比
    # ----------------------
    plt.axhline(ensemble_dice, color=ENSEMBLE_COLORS[0], linewidth=3,
                label=f'Ensemble ({ensemble_dice:.3f})')

    # ----------------------
    # 样式优化
    # ----------------------
    plt.title(f"{model_name} Performance Comparison\n"
              f"5-Fold Mean: {mean:.3f}±{std:.3f} | "
              f"Ensemble: {ensemble_dice:.3f}", pad=20)

    plt.xticks(range(len(folds)), [f'Fold {i + 1}' for i in range(len(folds))])
    plt.ylabel("Dice Coefficient")
    plt.ylim(min(folds) * 0.95, max(ensemble_dice, max(folds)) * 1.01)  # 动态调整Y轴范围

    # 添加统计结论
    conclusion = ("Ensemble outperforms all folds!"
                  if ensemble_dice > max(folds) else
                  "️ Ensemble not better than best fold")
    plt.annotate(conclusion, xy=(0.5, 0.92), xycoords='axes fraction',
                 ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', fc='white'))

    # 调整图例位置
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=len(folds) + 2)
    plt.grid(axis='y', alpha=0.3)

    plt.savefig(f"{model_name}_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


# ------------------------------
# 箱型图函数
# ------------------------------
def plot_boxplot(all_models):
    plt.figure(figsize=(8, 6))
    box_data = [[item[1] for item in model["folds"]] for model in all_models.values()]
    labels = [f"{name}\n(n={len(data['folds'])} folds)"
              for name, data in all_models.items()]

    box = plt.boxplot(box_data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(box['boxes'], ['#1f77b4', '#ff7f0e']):
        patch.set_facecolor(color)

    # 标注集成模型位置
    ensemble_handles = []
    for i, (model_name, model) in enumerate(all_models.items()):
        ensemble_dice = model["ensemble"]
        scatter = plt.scatter(i + 1, ensemble_dice, color=ENSEMBLE_COLORS[i], s=100, edgecolor='white', zorder=3)
        ensemble_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=ENSEMBLE_COLORS[i], markersize=10, label=f'{model_name} Ensemble'))

    plt.title("Fold Performance Distribution with Ensemble Markers")
    plt.ylabel("Dice Coefficient")

    # 创建自定义图例
    box_handles = [Line2D([0], [0], color='w', markerfacecolor=color, marker='s', markersize=10, label=label) for color, label in zip(['#1f77b4', '#ff7f0e'], labels)]
    all_handles = box_handles + ensemble_handles
    plt.legend(handles=all_handles, loc='upper right')

    plt.tight_layout()
    plt.savefig("boxplot_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


# ------------------------------
# 条形图函数
# ------------------------------
def plot_barplots(all_models):
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))  # 修改为上下结构，调整图形大小

    for i, (model_name, model_data) in enumerate(all_models.items()):
        model_folds = [item[1] for item in model_data["folds"]]
        model_ensemble = model_data["ensemble"]
        model_mean = np.mean(model_folds)
        model_std = np.std(model_folds)

        x = np.arange(len(model_folds) + 1)
        width = 0.35

        colors = FOLD_COLORMAP(np.linspace(0, 1, len(model_folds)))
        rects1 = axes[i].bar(x[:-1], model_folds, width, color=colors, label=[f'Fold {j + 1}' for j in range(len(model_folds))])
        rects2 = axes[i].bar(x[-1], model_ensemble, width, color=ENSEMBLE_COLORS[0], label='Ensemble')

        axes[i].errorbar(x[:-1], model_folds, yerr=model_std, fmt='none', ecolor='black', capsize=5)
        axes[i].set_ylabel('Dice Coefficient')
        axes[i].set_title(f'{model_name} Performance')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([f'Fold {j + 1}' for j in range(len(model_folds))] + ['Ensemble'])

        # 在每个bar上添加数值
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                axes[i].annotate('{:.3f}'.format(height),
                                 xy=(rect.get_x() + rect.get_width() / 2, height),
                                 xytext=(0, 3),  # 3 points vertical offset
                                 textcoords="offset points",
                                 ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        # 调整图例位置
        axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=len(model_folds) + 1)

    plt.tight_layout()
    plt.savefig("barplots_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


# ------------------------------
# 主程序调用部分
# ------------------------------
if __name__ == "__main__":
    # 准备测试数据
    all_models = {
        "ModelA": {
            "folds": [(i, val) for i, val in enumerate([0.89, 0.85, 0.87, 0.84, 0.86])],
            "ensemble": 0.88
        },
        "ModelB": {
            "folds": [(i, val) for i, val in enumerate([0.82, 0.83, 0.81, 0.80, 0.84])],
            "ensemble": 0.85
        }
    }

    # 同时生成三种可视化
    for model_name in all_models:
        plot_model_comparison(all_models[model_name], model_name)  # 原有功能：单个模型图

    plot_boxplot(all_models)  # 新增功能：箱型图
    plot_barplots(all_models)  # 新增功能：条形图