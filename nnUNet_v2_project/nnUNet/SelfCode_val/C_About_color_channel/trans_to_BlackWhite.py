import cv2
import os
import numpy as np

# 输入文件夹（包含二值分割 PNG 图片）
input_folder = r"C:\Users\lenovo\nnUNet\output"  # 使用原始字符串
output_folder = r"C:\Users\lenovo\nnUNet\output_final"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历所有 PNG 文件
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):  # 只处理 PNG 图片
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 读取二值图
        binary_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # 转换为黑白图（0->黑, 1->白 255）
        colored_mask = (binary_mask * 255).astype(np.uint8)

        # 保存处理后的图片
        cv2.imwrite(output_path, colored_mask)

print("✅ 所有图片已转换完成！")