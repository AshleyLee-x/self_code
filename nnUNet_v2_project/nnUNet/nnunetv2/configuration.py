#设置默认的并行处理数量，并定义一个用于判断样本各项异性的阈值
import os

from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
###设置默认进程数量
default_num_processes = 8 if 'nnUNet_def_n_proc' not in os.environ else int(os.environ['nnUNet_def_n_proc'])
###设置各项异性阈值( 设置为 3，表示当样本在低分辨率轴上的间距是下一个较大间距的 3 倍时，样本被视为各项异性)
ANISO_THRESHOLD = 3  # determines when a sample is considered anisotropic (3 means that the spacing in the low
# resolution axis must be 3x as large as the next largest spacing)
###获取允许的处理数量
default_n_proc_DA = get_allowed_n_proc_DA()
