#set dataset doc path

import os

nnUNet_raw = 'C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_raw'
nnUNet_preprocessed = 'C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_preprocessed'
nnUNet_results = 'C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_results'

if nnUNet_raw is None:
    print("nnUNet_raw is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if nnUNet_preprocessed is None:
    print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")

if nnUNet_results is None:
    print("nnUNet_results is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")
