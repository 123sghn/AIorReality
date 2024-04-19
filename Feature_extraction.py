# 特征提取的4种方式
import cv2
import numpy as np
from sklearn.decomposition import PCA
from skimage.feature import hog
import torch
import torchvision.transforms as transforms
import torchvision.models as models


# 特征提取方式一：像素直方图
def get_hist(origin_data):
    temp_data = []
    for i in origin_data:
        # 读取图像
        image = cv2.imdecode(np.fromfile(i, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 为方便处理，统一将图像像素大小调整为256x256
        img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        # 计算图像直方图并存储至X数组
        hist = cv2.calcHist([img], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
        # 归一化并展平
        temp_data.append(((hist / 255).flatten()))
    return temp_data


# 特征提取方式二：提取像素直方图并进行PCA降维
def get_pca_features(origin_data, n_components=100):
    pca = PCA(n_components=n_components)
    features = []
    for i in origin_data:
        # 读取图像
        image = cv2.imdecode(np.fromfile(i, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 为方便处理，统一将图像像素大小调整为256x256
        img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        # 计算图像直方图
        hist = cv2.calcHist([img], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
        # 归一化
        hist_normalized = (hist / 255).flatten()
        features.append(hist_normalized)
    # 使用PCA进行特征提取
    pca.fit(features)
    return pca.transform(features)


# 特征提取方式三：提取方向梯度直方图
def get_hog_features(origin_data, cell_size=(8, 8), block_size=(2, 2), bins=9):
    hog_features = []
    for i in origin_data:
        # 读取图像
        image = cv2.imdecode(np.fromfile(i, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 为方便处理，统一将图像像素大小调整为256x256
        img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        # 使用HOG特征提取器提取HOG特征
        hog_feature = hog(
            img,
            orientations=bins,
            pixels_per_cell=cell_size,
            cells_per_block=block_size,
            block_norm="L2-Hys",
            visualize=False,
            multichannel=True,
        )
        hog_features.append(hog_feature)
    return hog_features


# 特征提取方式四：CNN
def get_cnn_features(origin_data, image_size=(256, 256)):
    # 加载预训练的ResNet模型，去掉顶层全连接层
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()

    # 图像预处理
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    features = []
    for i in origin_data:
        img = cv2.imdecode(np.fromfile(i, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            feature = model(img)

        feature = feature.squeeze().numpy()  # 将张量转换为NumPy数组
        feature = feature.flatten()  # 将特征向量展平为一维向量
        features.append(feature)

    features = np.array(features)  # 转换为NumPy数组
    return features
