from PIL import Image
import os

# 图像文件夹路径
image_folder = 'C:\\nnUNet_v2_project\\nnUNet\\DATASET\\nnUNet_raw\\Dataset666_Cells\\imagesTr'

# 遍历图像文件
for filename in os.listdir(image_folder):
    if filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)
        num_channels = image.mode
        if num_channels == 'RGBA':
            print(f'{filename} 有 4 个通道 (RGBA)')
        elif num_channels == 'RGB':
            print(f'{filename} 有 3 个通道 (RGB)')
        else:
            print(f'{filename} 通道数: {num_channels}')