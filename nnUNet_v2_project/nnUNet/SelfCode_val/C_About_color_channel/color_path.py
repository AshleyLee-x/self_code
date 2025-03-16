train_input_dir = 'C:/nnUNet_v2_project/nnUNet_v2/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/cells/test/images'  # 训练集图像目录
test_input_dir = 'C:/nnUNet_v2_project/nnUNet_v2/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/cells/train/images'  # 测试集图像目录
output_dir = 'C:/nnUNet_v2_project/nnUNet_v2/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/cells/color_channel_pre_result'  # 输出目录
main(train_input_dir, test_input_dir, output_dir)