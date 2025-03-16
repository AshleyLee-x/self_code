import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 定义颜色方案
COLORS = {
    "HSV": ("#1f77b4", "#08306b"),  # 蓝色系（浅，深）
    "RGB": ("#2ca02c", "#00441b"),  # 绿色系（浅，深）
    "Grayscale": ("#ff7f0e", "#8B0000")  # 橙色/红色系（浅，深）
}


def load_model_data(file_path):
    """加载单个模型数据并计算指标"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 处理每个案例的指标
    case_metrics = []
    for case in data["metric_per_case"]:
        m = case["metrics"]["1"]
        tp, fp, fn = m["TP"], m["FP"], m["FN"]

        # 计算精确率和召回率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        case_metrics.append({
            "Dice": m["Dice"],
            "Precision": precision,
            "Recall": recall
        })

    # 处理前景类别均值
    fg = data["foreground_mean"]
    tp, fp, fn = fg["TP"], fg["FP"], fg["FN"]
    fg_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fg_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "cases": case_metrics,
        "foreground": {
            "Dice": fg["Dice"],
            "Precision": fg_precision,
            "Recall": fg_recall
        }
    }


def plot_metric(metric, model_data, colors):
    """绘制单个指标对比图"""
    plt.figure(figsize=(10, 6))

    for model_name, data in model_data.items():
        # 获取案例数据和前景均值
        values = [case[metric] for case in data["cases"]]
        fg_value = data["foreground"][metric]

        # 生成x轴坐标（案例索引+前景均值位置）
        x = np.arange(len(values))
        fg_x = len(values)  # 均值点放在最后

        # 绘制案例折线
        plt.plot(x, values,
                 color=colors[model_name][0],
                 marker='o',
                 label=f'{model_name} Cases')

        # 突出显示前景均值点
        plt.scatter(fg_x, fg_value,
                    color=colors[model_name][1],
                    s=100, zorder=5,
                    label=f'{model_name} Foreground')

    plt.title(f'{metric} Comparison', fontsize=14)
    plt.xlabel('Case Index', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.grid(True, alpha=0.3)

    # 合并重复图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               loc='upper right', bbox_to_anchor=(1, 0.95))

    plt.tight_layout()
    plt.savefig(f'{metric}_comparison.png', dpi=300)
    plt.close()


# 加载所有模型数据
model_files = {
    "HSV": "HSV_summary.json",
    "RGB": "RGB_summary.json",
    "Grayscale": "grayscale_summary.json"
}

model_data = {}
for model, path in model_files.items():
    model_data[model] = load_model_data(path)

# 生成所有对比图
for metric in ["Dice", "Precision", "Recall"]:
    plot_metric(metric, model_data, COLORS)