import cv2
import os
from tqdm import tqdm


def convert_to_grayscale(input_folder: str, output_folder: str):
    """
    将指定文件夹中的所有PNG图片转换为灰度图
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有PNG文件
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    print(f"找到 {len(png_files)} 个PNG文件")

    # 处理每个文件
    for filename in tqdm(png_files, desc="转换进度"):
        # 读取图片
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        # 转换为灰度图
        if img is not None:
            if len(img.shape) > 2:  # 如果是彩色图
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # 保存灰度图
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, gray)
        else:
            print(f"无法读取图片: {filename}")


if __name__ == "__main__":
    # 直接指定输入输出路径
    input_folder = r"C:\Users\lenovo\nnUNet\Dataset2\training\input"
    output_folder = r"C:\Users\lenovo\nnUNet\Dataset3\training\input"

    print("开始转换图片...")
    convert_to_grayscale(input_folder, output_folder)
    print("✅ 转换完成！")