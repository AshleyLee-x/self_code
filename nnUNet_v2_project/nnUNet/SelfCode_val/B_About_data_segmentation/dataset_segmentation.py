import os
import random
import shutil


def split_data():
    # Set the ratio of training set to test set!!!!train:test7:3
    train_ratio = 0.7

    # Set the paths to the image and label folders
    image_folder = 'C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/image_data'
    label_folder = 'C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/image_label'

    # Create train and test folders
    os.makedirs('C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/UseData_cells/train/images', exist_ok=True)
    os.makedirs('C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/UseData_cells/train/labels', exist_ok=True)
    os.makedirs('C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/UseData_cells/test/images', exist_ok=True)
    os.makedirs('C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/UseData_cells/test/labels', exist_ok=True)

    # Get the list of image files
    image_files = os.listdir(image_folder)
    # Shuffle the list randomly
    random.shuffle(image_files)

    # Calculate the number of images for training and testing
    num_train = int(len(image_files) * train_ratio)
    num_test = len(image_files) - num_train

    # Copy the images and labels to the respective train and test folders
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        label_file = image_file.split('.')[0] + '.png'
        label_path = os.path.join(label_folder, label_file)

        if i < num_train:
            # Copy to train folder
            shutil.copy(image_path, 'C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/UseData_cells/train/images')
            shutil.copy(label_path, 'C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/UseData_cells/train/labels')
        else:
            # Copy to test folder
            shutil.copy(image_path, 'C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/UseData_cells/test/images')
            shutil.copy(label_path, 'C:/nnUNet_v2_project/nnUNet/DATASET/nnUNet_raw/setfor_nnunet/UseData_cells/test/labels')


if __name__ == '__main__':
    split_data()