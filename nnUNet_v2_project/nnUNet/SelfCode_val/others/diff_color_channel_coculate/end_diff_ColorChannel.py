import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

# 定义颜色方案
COLORS = {
    "HSV": ("#1f77b4", "#08306b"),
    "RGB": ("#2ca02c", "#00441b"),
    "Grayscale": ("#ff7f0e", "#8B0000")
}


def cliffs_delta(x, y):
    """计算Cliff's delta效应量"""
    n_x, n_y = len(x), len(y)
    count = 0
    for i in x:
        for j in y:
            if i > j:
                count += 1
            elif i < j:
                count -= 1
    return count / (n_x * n_y)


def load_model_data(file_path):
    """加载并处理模型数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    case_metrics = []
    for case in data["metric_per_case"]:
        m = case["metrics"]["1"]
        tp, fp, fn = m["TP"], m["FP"], m["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        case_metrics.append({
            "Dice": m["Dice"],
            "Precision": precision,
            "Recall": recall,
            "IoU": m["IoU"]
        })

    # 处理前景均值
    fg = data["foreground_mean"]
    tp, fp, fn = fg["TP"], fg["FP"], fg["FN"]
    return {
        "cases": case_metrics,
        "foreground": {
            "Dice": fg["Dice"],
            "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "Recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "IoU": fg["IoU"]
        }
    }


def generate_stat_table(model_data, metric):
    """生成统计结果表格"""
    models = list(model_data.keys())
    comparisons = []

    # 执行两两比较
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            data1 = [case[metric] for case in model_data[m1]["cases"]]
            data2 = [case[metric] for case in model_data[m2]["cases"]]

            # Wilcoxon检验
            _, p = wilcoxon(data1, data2)
            d = cliffs_delta(data1, data2)
            n_comp = len(models) * (len(models) - 1) // 2  # 比较次数
            adj_p = min(p * n_comp, 1.0)  # Bonferroni校正

            comparisons.append({
                "Comparison": f"{m1} vs {m2}",
                "Median1": f"{np.median(data1):.3f}",
                "Median2": f"{np.median(data2):.3f}",
                "p-value": f"{adj_p:.4f}{'***' if adj_p < 0.001 else '**' if adj_p < 0.01 else '*' if adj_p < 0.05 else ''}",
                "Effect": f"{d:.2f}"
            })

    # 创建表格
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    table = ax.table(
        cellText=[[c["Comparison"], c["Median1"], c["Median2"], c["p-value"], c["Effect"]]
                  for c in comparisons],
        colLabels=["Comparison", "Median1", "Median2", "Adj p-value", "Cliff's δ"],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.savefig(f'{metric}_stats.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric(metric, model_data, colors):
    """绘制带统计验证的对比图"""
    plt.figure(figsize=(12, 6))

    # 收集案例数据
    case_data = {model: [case[metric] for case in data["cases"]]
                 for model, data in model_data.items()}

    # 绘制每个模型
    for model, data in model_data.items():
        values = case_data[model]
        fg_value = data["foreground"][metric]
        x = np.arange(len(values))

        # 绘制案例折线
        plt.plot(x, values, color=colors[model][0], marker='o',
                 markersize=6, linewidth=1.5, alpha=0.8, label=f'{model} Cases')

        # 添加前景均值误差线（IQR）
        q1, q3 = np.percentile(values, [25, 75])
        plt.errorbar(x[-1] + 0.5, fg_value, yerr=(q3 - q1) / 2,
                     fmt='o', color=colors[model][1], markersize=10,
                     elinewidth=2, capsize=5, label=f'{model} FG')

        # 标记离群案例（超过1.5IQR）
        upper_bound = q3 + 1.5 * (q3 - q1)
        lower_bound = q1 - 1.5 * (q3 - q1)
        for idx, v in enumerate(values):
            if v > upper_bound or v < lower_bound:
                plt.text(idx, v, '*', color='red', ha='center',
                         va='bottom', fontsize=14, weight='bold')

    # 图表装饰
    plt.title(f'{metric} Performance Comparison\n*: Significant outliers (1.5IQR)', pad=20)
    plt.xlabel('Case Index', labelpad=10)
    plt.ylabel(metric, labelpad=10)
    plt.grid(alpha=0.3)
    plt.xticks(np.arange(0, len(values) + 1, 5))

    # 合并图例
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:3], labels[:3], ncol=3,
               loc='upper center', bbox_to_anchor=(0.5, -0.15),
               frameon=False)

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(f'{metric}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # 加载数据
    model_files = {
        "HSV": "HSV_summary.json",
        "RGB": "RGB_summary.json",
        "Grayscale": "grayscale_summary.json"
    }
    model_data = {model: load_model_data(path) for model, path in model_files.items()}

    # 生成图表和统计表
    for metric in ["Dice", "Precision", "Recall", "IoU"]:
        plot_metric(metric, model_data, COLORS)
        generate_stat_table(model_data, metric)