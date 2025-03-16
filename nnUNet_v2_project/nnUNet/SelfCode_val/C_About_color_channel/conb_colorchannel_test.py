import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score, dice_score


# 图像读取
def load_image(image_path):
    image = cv2.imread(image_path)
    return image


# 转换颜色空间
def convert_color_spaces(image):
    # RGB -> L*a*b
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # RGB -> YCbCr
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # 提取L*a*b通道
    lab_a = lab_image[:, :, 1]
    lab_b = lab_image[:, :, 2]

    # 提取YCbCr通道
    ycbcr_cb = ycbcr_image[:, :, 1]
    ycbcr_cr = ycbcr_image[:, :, 2]

    return lab_a, lab_b, ycbcr_cb, ycbcr_cr


# 图像预处理：高斯平滑
def preprocess_image(image):
    smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
    return smoothed_image


# k-means分割
def kmeans_segmentation(image, k):
    image_flattened = image.reshape((-1, 1))  # 扁平化
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image_flattened)
    segmented_image = kmeans.labels_.reshape(image.shape)
    return segmented_image


# 多通道叠加
def multi_channel_stack(lab_a, lab_b, ycbcr_cb, ycbcr_cr):
    # 叠加L*a*b的a、b通道与YCbCr的Cb、Cr通道
    stacked_channels = np.stack((lab_a, lab_b, ycbcr_cb, ycbcr_cr), axis=-1)
    return stacked_channels


# 评估分割性能（示例：Dice系数）
def evaluate_segmentation(true_labels, predicted_labels):
    dice = dice_score(true_labels.flatten(), predicted_labels.flatten())
    return dice
# 主程序
def main(image_path):
    # 1. 读取图像
    image = load_image(image_path)
    # 2. 图像预处理（高斯平滑）
    smoothed_image = preprocess_image(image)
    # 3. 转换颜色空间
    lab_a, lab_b, ycbcr_cb, ycbcr_cr = convert_color_spaces(smoothed_image)
    # 4. 多通道叠加
    stacked_image = multi_channel_stack(lab_a, lab_b, ycbcr_cb, ycbcr_cr)
    # 5. 使用k-means进行图像分割
    k = 4  # 设置k值（例如4）
    segmented_image = kmeans_segmentation(stacked_image, k)
    # 6. 评估分割结果
    # 假设我们有真实标签（true_labels），这里用伪代码表示
    true_labels = np.zeros_like(segmented_image)  # 真实标签（此处仅为示例）
    dice_score = evaluate_segmentation(true_labels, segmented_image)
    print(f"Dice Score: {dice_score:.4f}")
# 运行主程序
if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"
    main(image_path)
