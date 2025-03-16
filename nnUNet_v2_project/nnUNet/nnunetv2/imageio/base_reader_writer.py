from abc import ABC, abstractmethod
from typing import Tuple, Union, List
import numpy as np


class BaseReaderWriter(ABC):
    @staticmethod
    def _check_all_same(input_list):
        if len(input_list) == 1:
            return True
        else:
            # compare all entries to the first
            return np.allclose(input_list[0], input_list[1:])

    @staticmethod
    def _check_all_same_array(input_list):
        # compare all entries to the first
        for i in input_list[1:]:
            if i.shape != input_list[0].shape or not np.allclose(i, input_list[0]):
                return False
        return True

    @abstractmethod
    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
    """
    read_images 方法用于读取一系列图像，并返回一个四维的 np.ndarray 以及一个字典。
    要求：
    1.数组结构:
    第一个轴表示不同的模态（或颜色通道，具体可自定）。
    形状为 (c, x, y, z)
    其中 c 是模态数量（可以为 1），x、y 和 z 是空间维度。
    2.字典内容:
    用于存储在转换为 NumPy 数组时丢失的必要元信息
    例如图像的间距、方向和方位。
    这个字典将传递给 write_seg 方法，以导出预测的分割结果，需要确保包含所有必要信息。
    3.重要说明
    字典中必须包含一个 spacing 键，其值为长度为 3 的元组或列表，表示 np.ndarray 的体素间距。
    例如：my_dict = {'spacing': (3, 0.5, 0.5), ...}。
    数字的顺序必须与返回的 NumPy 数组的轴顺序相对应：
    如果数组形状为 (c, x, y, z)，则 (a, b, c) 应对应 x、y 和 z 的间距。
    4.2D 图像处理:
    返回的数组应具有形状 (c, 1, x, y)，间距应为 (999, sp_x, sp_y)。确保 999 大于 sp_x 和 sp_y！
    例如：形状为 (3, 1, 224, 224)，间距为 (999, 1, 1)。
    5.无间距的图像:
    设置间距为 1（2D 图像的例外情况仍适用）。
    """
        pass

    @abstractmethod
    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
    """
     1.返回的分割结果必须具有形状 1,x,y,z(必须是四维的,其中第一个维度为 1，后面三个维度对应空间尺寸。)
     2.目前不支持同时返回多个分割结果。
     3.如果图像和分割的读取方式相同，可以直接调用 read_image 方法。
     4.参数说明： seg_fname: 分割文件的名称。
     5.返回说明：(1)返回一个形状为 (1, x, y, z) 的 np.ndarray
               其中 x, y, z 是空间维度（对于 2D 图像，x 设置为 1；例如，对于 2D 分割，形状为 (1, 1, 224, 224)）。
               (2)返回一个包含元数据的字典。
               字典可以包含任何信息，但必须包括 {'spacing': (a, b, c)}:
               a 是 x 的间距，
               b 是 y 的间距，
               c 是 z 的间距。
               如果图像没有间距，则设置为 1。
               对于 2D 图像，a 设置为 999（最大间距值，确保大于 b 和 c）。
    """
        pass

    @abstractmethod
    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        """
        1.将预测的分割结果导出到所需的文件格式, 输入的分割数组与图像数据在形状和方向上是一致的，因此无需进行额外处理。
        2.properties 是在 read_images/read_seg 时创建的相同字典，因此可以在这里使用该信息来恢复元数据。

        IMPORTANT: 此方法假设所有分割结果都是三维的;对于 2D 图像，分割结果的形状为 (1, y, z)。
                   (如果输入是 2D 图像，需要通过 seg = seg[0] 将 3D 分割转换为 2D。)
        -seg: 一个分割数组，类型为 np.ndarray，元素为整数，形状为 (x, y, z)。对于 2D 分割，形状为 (1, y, z)。
        -output_fname: 输出文件的名称。
        -properties: 在 read_images 时创建的字典，包含与此分割相关的元数据，用于恢复元数据。
        :return:
        """
        pass