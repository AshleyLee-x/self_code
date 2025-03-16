import os
from skimage import io
import numpy as np

# 定义结果文件夹路径
result_folder = 'C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_results/Dataset777_Cells/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessed'
# 定义输出文件夹路径
output_folder = 'C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_results/Dataset777_Cells/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessed_result'

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历结果文件夹中的所有文件
for filename in os.listdir(result_folder):
    if filename.endswith('.png'):
        # 构建完整的文件路径
        input_path = os.path.join(result_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 读取标签图像
        label = io.imread(input_path)

        # 将标签值放大，使前景（值为 1）变为 255
        enhanced_label = label * 255

        # 确保数据类型为 uint8
        enhanced_label = enhanced_label.astype(np.uint8)

        # 保存增强后的标签图像
        io.imsave(output_path, enhanced_label, check_contrast=False)

        print(f"Converted {filename} to visible format and saved to {output_path}")