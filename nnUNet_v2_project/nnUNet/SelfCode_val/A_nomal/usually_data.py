#建议对环境、库、包管理在anaconda prompt中进行，对项目管理都在pycharm中进行

##conda開啓對應環境
###語句：
conda activate [環境名]
###eg:
conda activate nnU-Net

##對於數據集合的轉化：
###关于语句的解释：nnUNet\documentation\convert_msd_dataset.md
###語句：
nnUNetv2_convert_MSD_dataset -i [data_path]
###eg:
nnUNetv2_convert_MSD_dataset -i D:\DeepLearning\nnUNet\nnUNetFrame\nnUNet_raw\Task04_Hippocampus

##數據集的預處理語句
###語句：
nnUNetv2_plan_and_preprocess -d n --verify_dataset_integrity
###eg:
nnUNetv2_plan_and_preprocess -d 004 --verify_dataset_integrity
###运行后生成的文件中：nnUNet\DATASET\nnUNet_preprocessed\Dataset004_Hippocampus\gt_segmentations是数据集的标签
###其中.npz文件是一个numpy格式的压缩文件，所以由numpy打开(打开方式可见：doc_npz_open.py)


##訓練
##包括三种U-Net网络配置
##分别是2D U-Net，3D全分辨率U-Net，3D U-Net级联（包括3D低分辨率U-Net和3D全分辨率U-Net）
##要进行级联下的3D全分辨率U-Net需先完成3D低分辨率U-Net
###注意使用cpu进行训练的语句是：
###一般例句
nnUNetv2_train [dataset_name] 2d [number_of_training] -[device_type]
###eg：
nnUNetv2_train 4 2d 0 -device cpu
####usually number of training is 5


nnUNetv2_plan_and_preprocess -d 777 --verify_dataset_integrity

