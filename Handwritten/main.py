import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2  # 导入OpenCV库
import os  # 导入os库以遍历文件夹

# 模型保存路径
model_path = "saved_model/mnist_model.h5"

# 加载模型
model = tf.keras.models.load_model(model_path)
print("模型已加载。")

def preprocess_image(image_path):
    """
    预处理输入图像，使其符合模型的输入要求，包括二值化和黑白反转
    :param image_path: 输入图像的路径
    :return: 预处理后的图像数据
    """
    # 打开图像并转换为灰度模式
    image = Image.open(image_path).convert('L')  # 'L'表示灰度图
    image = image.resize((28, 28))  # 调整图像大小为28x28

    # 将图像转换为numpy数组
    image_array = np.array(image)

    # 应用二值化处理，使用OpenCV的Otsu阈值方法
    _, binary_image = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 黑白反转处理：将0变为1，将1变为0
    inverted_image = 255 - binary_image

    # 归一化并调整形状
    inverted_image = inverted_image / 255.0
    inverted_image = inverted_image.reshape(1, 28, 28, 1)  # 调整形状为(1, 28, 28, 1)

    return inverted_image

def predict_images_in_folder(folder_path):
    """
    读取文件夹中的所有图像，并对每张图像进行预测
    :param folder_path: 包含图像的文件夹路径
    """
    # 获取文件夹中的所有文件名
    for file_name in os.listdir(folder_path):
        # 构建图像文件的完整路径
        image_path = os.path.join(folder_path, file_name)

        # 仅处理图像文件（可以根据需要调整文件格式，如 .jpg, .png 等）
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                # 预处理图像
                processed_image = preprocess_image(image_path)

                # 进行预测
                prediction = model.predict(processed_image)
                predicted_class = prediction.argmax()

                # 输出预测结果
                print(f"文件: {file_name} -> 预测结果: {predicted_class}")

                # 可视化图像及预测
                # plt.imshow(processed_image.reshape(28, 28), cmap='gray')
                # plt.title(f"文件: {file_name}, 预测: {predicted_class}")
                # plt.show()
            except Exception as e:
                print(f"无法处理图像 {file_name}: {e}")

# 设置包含图像的文件夹路径
folder_path = "D:\Study\图像处理技术及其在声呐图像中的应用\Handwritten\images"  # 将此路径替换为你图像文件夹的路径

# 批量预测文件夹中的图像
predict_images_in_folder(folder_path)
