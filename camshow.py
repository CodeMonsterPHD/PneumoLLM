import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 加载预训练的ViT模型和特征提取器
model_name = "google/vit-base-patch16-224-in21k"
model_config_path = "G:/temp/config.json"  # 替换为你的配置文件路径
import json
from transformers import ViTConfig
config = ViTConfig.from_json_file(model_config_path)
model = ViTForImageClassification(config)
feature_extractor = ViTFeatureExtractor(config)

# 加载图像并进行预处理
image_path = "G:/temp/OIP.jpg"
image = Image.open(image_path)
image = feature_extractor(images=image, return_tensors="pt")["pixel_values"]

# 使用模型进行预测
with torch.no_grad():
    outputs = model(image)
predictions = torch.softmax(outputs.logits, dim=1)

# 获取预测的类别
predicted_class = torch.argmax(predictions).item()

# 获取模型的最后一个卷积层特征图
last_conv_layer = model.vit.encoder.layer[-1].output

# 反向传播以获取类别激活图
model.zero_grad()
with torch.set_grad_enabled(True):
    class_activation_map = torch.matmul(predictions, last_conv_layer.view(last_conv_layer.size(0), -1))
class_activation_map = class_activation_map.view(last_conv_layer.size(0), last_conv_layer.size(2), last_conv_layer.size(3))
class_activation_map = F.relu(class_activation_map).cpu().numpy()

# 根据CAM图生成热度图
heatmap = class_activation_map[0, predicted_class, :, :]
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

# 将热度图叠加到原始图像上
heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
heatmap = heatmap.resize((image.shape[3], image.shape[2]))
heatmap = heatmap.convert("RGB")
superimposed_img = Image.blend(image.squeeze().cpu(), heatmap, alpha=0.5)

# 显示原始图像和CAM图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image.squeeze().permute(1, 2, 0))
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(superimposed_img)
plt.title("CAM Image")
plt.show()
