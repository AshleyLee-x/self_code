import multiprocessing
import shutil
from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes

def load_and_convert_case(input_image: str, input_seg: str, output_image: str, output_seg: str):
    """
    加载并转换单个案例的图像和分割标签
    """
    seg = io.imread(input_seg)
    seg[seg == 255] = 1  # 二值化处理
    image = io.imread(input_image)

    io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_image, output_image)

if __name__ == "__main__":
    # 输入文件夹路径
    source = r'C:\Users\lenovo\nnunet\Dataset3'  # 替换为你的数据集路径
    dataset_id = 2  # 替换为你想要的数据集ID
    dataset_name = f'Dataset{dataset_id:03d}_PNG'

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
                         {0: 'R', 1: 'G', 2: 'B'},  # RGB通道
                         {'background': 0, 'foreground': 1},  # 二值分割标签
                         num_train, 
                         '.png')

    print("✅ 数据集转换完成！")