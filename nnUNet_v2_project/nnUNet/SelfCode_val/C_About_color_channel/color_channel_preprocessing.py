# 为了实现将每张.png图像的R、G、B通道分别保留，并将同一通道的结果收集在同一文件夹中，可以对原始代码进行以下修改：
# 创建用于保存每个通道的文件夹。
# 遍历数据集中的所有图像，提取R、G、B通道并保存到对应的文件夹中。
# 1.接受两个输入路径：一个用于训练集，另一个用于测试集。
# 2.为训练集和测试集分别创建输出文件夹，以便将结果组织得更加清晰。

import os
import multiprocessing
from skimage import io, exposure
import numpy as np


def enhance_contrast(image):
    # 使用直方图均衡化增强对比度
    return exposure.equalize_hist(image)


def extract_rgb_channels(input_image: str, output_r: str, output_g: str, output_b: str):
    try:
        # 读取图像
        image = io.imread(input_image)

        # 打印图像信息
        print(f"Processing {input_image}: shape={image.shape}, dtype={image.dtype}")

        # 处理 RGBA 图像
        if image.ndim == 3:
            if image.shape[2] == 4:  # RGBA
                r_channel = image[:, :, 0]
                g_channel = image[:, :, 1]
                b_channel = image[:, :, 2]
            elif image.shape[2] == 3:  # RGB
                r_channel = image[:, :, 0]
                g_channel = image[:, :, 1]
                b_channel = image[:, :, 2]
            else:
                raise ValueError(f"输入图像 {input_image} 不是有效的 RGB 或 RGBA 格式")
        elif image.ndim == 2:  # 灰度图像
            r_channel = image
            g_channel = image
            b_channel = image
        else:
            raise ValueError(f"输入图像 {input_image} 的维度不正确")

        # 增强对比度
        r_channel = enhance_contrast(r_channel)
        g_channel = enhance_contrast(g_channel)
        b_channel = enhance_contrast(b_channel)

        # 创建 RGB 图像以保存通道
        r_image = np.zeros_like(image)
        g_image = np.zeros_like(image)
        b_image = np.zeros_like(image)

        # 仅保留各自通道的值
        r_image[:, :, 0] = (r_channel * 255).astype(np.uint8)
        g_image[:, :, 1] = (g_channel * 255).astype(np.uint8)
        b_image[:, :, 2] = (b_channel * 255).astype(np.uint8)

        # 保存每个通道的图像
        io.imsave(output_r, r_image)
        io.imsave(output_g, g_image)
        io.imsave(output_b, b_image)
        print(f"Processed: {input_image}")
    except Exception as e:
        print(f"Error processing {input_image}: {e}")


def process_images(input_dir: str, output_dir: str):
    os.makedirs(os.path.join(output_dir, 'R'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'G'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'B'), exist_ok=True)

    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')]

    with multiprocessing.Pool() as pool:
        for image_path in image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_r = os.path.join(output_dir, 'R', f'{base_name}_r.png')
            output_g = os.path.join(output_dir, 'G', f'{base_name}_g.png')
            output_b = os.path.join(output_dir, 'B', f'{base_name}_b.png')

            pool.apply_async(extract_rgb_channels, (image_path, output_r, output_g, output_b))

        pool.close()
        pool.join()


def main(train_input_dir: str, test_input_dir: str, output_dir: str):
    train_output_dir = os.path.join(output_dir, 'train')
    process_images(train_input_dir, train_output_dir)

    test_output_dir = os.path.join(output_dir, 'test')
    process_images(test_input_dir, test_output_dir)


if __name__ == '__main__':
    train_input_dir = 'C:/nnUNet_v2_project/nnUNet_v2/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/cells/train/images'  # 训练集图像目录
    test_input_dir = 'C:/nnUNet_v2_project/nnUNet_v2/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/cells/test/images'  # 测试集图像目录
    output_dir = 'C:/nnUNet_v2_project/nnUNet_v2/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/cells/color_channel_pre_result'  # 输出目录
    main(train_input_dir, test_input_dir, output_dir)

# (1)代码说明:
# 1.提取R、G、B通道：extract_rgb_channels函数读取输入图像并提取R、G、B通道，分别保存为不同的文件。
# 2.处理多个图像：process_images函数遍历指定目录下的所有.png图像，并为每个通道创建对应的输出文件夹。
# 3.使用多进程处理：代码使用multiprocessing.Pool来并行处理多个图像，提高处理效率。
# (2)使用方法:
# 1.替换input_dir和output_dir为实际的输入和输出路径。
# 2.运行代码，程序将遍历输入目录下的所有PNG图像，提取RGB通道并将它们分别保存到对应的文件夹中。
