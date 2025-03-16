import multiprocessing
import shutil
from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
import cv2
import numpy as np

def load_and_convert_case(input_image: str, input_seg: str, output_image: str, output_seg: str):
    """
    加载并转换单个案例的RGB图像和分割标签
    """
    # 读取分割标签
    seg = io.imread(input_seg)
    seg[seg == 255] = 1  # 二值化处理

    # 读取RGB图像并转换为HSV
    image = io.imread(input_image)  # 读取RGB图像
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # 转换为HSV

    # 保存处理后的图像和标签
    io.imsave(output_seg, seg, check_contrast=False)
    io.imsave(output_image, image, check_contrast=False)

if __name__ == "__main__":
    # 输入文件夹路径
    source = r'C:\Users\lenovo\nnunet\Dataset2'  # 替换为你的数据集路径
    dataset_id = 4  # 替换为你想要的数据集ID
    dataset_name = f'Dataset{dataset_id:04d}_HSV'  # 修改名称以反映HSV图特性

    # 创建nnUNet标准目录
    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(source, 'training')
    test_source = join(source, 'testing')

    with multiprocessing.get_context("spawn").Pool(8) as p:
        # 处理训练集
        valid_ids = subfiles(join(train_source, 'output'), join=False, suffix='png')
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                        join(train_source, 'input', v),
                        join(train_source, 'output', v),
                        join(imagestr, v[:-4] + '_0000.png'),
                        join(labelstr, v),
                    ),)
                )
            )

        # 处理测试集
        valid_ids = subfiles(join(test_source, 'output'), join=False, suffix='png')
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                        join(test_source, 'input', v),
                        join(test_source, 'output', v),
                        join(imagests, v[:-4] + '_0000.png'),
                        join(labelsts, v),
                    ),)
                )
            )
        _ = [i.get() for i in r]

    # 生成dataset.json
    generate_dataset_json(join(nnUNet_raw, dataset_name),
                         {0: 'H', 1: 'S', 2: 'V'},  # HSV图像的3个通道
                         {'background': 0, 'foreground': 1},  # 二值分割标签
                         num_train,
                         '.png')

    print("✅ HSV图数据集转换完成！")