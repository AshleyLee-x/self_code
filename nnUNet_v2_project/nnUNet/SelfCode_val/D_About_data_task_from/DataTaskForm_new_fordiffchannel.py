import multiprocessing
import shutil
from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes
import numpy as np


def load_and_convert_case(input_image: str, input_seg: str, output_image: str, output_seg: str,
                          min_component_size: int = 50):
    try:
        seg = io.imread(input_seg)
        seg[seg == 255] = 1
        image = io.imread(input_image)
        print(f"Before conversion, image shape: {image.shape}")  # 打印转换前的图像形状
        if image.shape[-1] == 4:
            # 如果是4通道图像，只取前3个通道
            image = image[..., :3]
        print(f"After conversion, image shape: {image.shape}")  # 打印转换后的图像形状

        mask = image.sum(2) == (3 * 255)  # 这里可以根据实际情况修改生成掩码的逻辑
        # the dataset has large white areas in which road segmentations can exist but no image information is available.
        # Remove the road label in these areas
        mask = generic_filter_components(mask, filter_fn=lambda ids, sizes: [i for j, i in enumerate(ids) if
                                                                             sizes[j] > min_component_size])
        mask = binary_fill_holes(mask)
        seg[mask] = 0
        io.imsave(output_seg, seg, check_contrast=False)

        # 转换数据类型为 uint8
        if image.dtype != np.uint8:
            if image.max() > 255:
                image = (image / (image.max() / 255)).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        io.imsave(output_image, image, check_contrast=False)
        saved_image = io.imread(output_image)
        print(f"Saved image shape: {saved_image.shape}")
    except Exception as e:
        print(f"Error processing {input_image}: {e}")


if __name__ == "__main__":
    source = 'C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/UseData_cells'
    print(f"Source data path: {source}")

    dataset_name = 'Dataset777_Cells'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    print(f"imagesTr path: {imagestr}")
    print(f"imagesTs path: {imagests}")
    print(f"labelsTr path: {labelstr}")
    print(f"labelsTs path: {labelsts}")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(source, 'train')
    test_source = join(source, 'test')
    print(f"Train source path: {train_source}")
    print(f"Test source path: {test_source}")

    valid_ids_train = subfiles(join(train_source, 'labels'), join=False, suffix='png')
    print(f"Number of training labels: {len(valid_ids_train)}")

    valid_ids_test = subfiles(join(test_source, 'labels'), join=False, suffix='png')
    print(f"Number of test labels: {len(valid_ids_test)}")

    with multiprocessing.get_context("spawn").Pool(8) as p:
        num_train = len(valid_ids_train)
        r = []
        for v in valid_ids_train:
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         join(train_source, 'images', v),
                         join(train_source, 'labels', v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v),
                         50
                     ),)
                )
            )

        for v in valid_ids_test:
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         join(test_source, 'images', v),
                         join(test_source, 'labels', v),
                         join(imagests, v[:-4] + '_0000.png'),
                         join(labelsts, v),
                         50
                     ),)
                )
            )
        _ = [i.get() for i in r]

    try:
        generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'histocyte': 1},
                              num_train, '.png', dataset_name=dataset_name)
    except Exception as e:
        print(f"Error generating dataset JSON: {e}")