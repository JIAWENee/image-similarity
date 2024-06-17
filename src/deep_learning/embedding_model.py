import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from transformers import SwinModel
from transformers import CLIPProcessor, CLIPModel


class SwinTransformer:
    def __init__(self):
        # 加载预训练的Swin Transformer模型和特征提取器
        model_path = "../../models/swin-base-patch4-window7-224"
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.model = SwinModel.from_pretrained(model_path)

    def extract_features(self, image_path):
        # 加载并预处理图片
        image = Image.open(image_path).convert('RGB')  # 如果输入图像是灰度图（单通道）或有透明通道的图像（如 RGBA），转换为 RGB 可以避免维度不匹配的问题。
        inputs = self.processor(images=image, return_tensors="pt")

        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)


class VisionTransformer:
    def __init__(self):
        model_path = "../../models/vit-base-patch16-224-in21k"
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.model = ViTModel.from_pretrained(model_path)

    def extract_features(self, image_path):
        # 加载并预处理图片
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")

        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)


class CLIP:
    def __init__(self):
        # 加载预训练的CLIP模型和处理器
        model_path = "../../models/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)

    def extract_features(self, image_path):
        # 加载并预处理图片
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")

        # 提取特征
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features


embedding_model_classes = {
    "swin_transformer": SwinTransformer,
    "vision_transformer": VisionTransformer,
    "clip": CLIP,
}
