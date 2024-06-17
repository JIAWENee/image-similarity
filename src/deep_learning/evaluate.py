from typing import List
import torch

from src.deep_learning.embedding_model import embedding_model_classes


def cosine_similarity(vector1, vector2):
    """计算余弦相似度"""
    return torch.nn.functional.cosine_similarity(vector1, vector2).item()


def evaluation(img_path_1: str, img_path_2: str, embedding_models: List[str]):
    """使用深度学习算法评估图片相似度"""
    output_dict = {}

    for model_name in embedding_models:
        # 提取图片特征
        model = embedding_model_classes[model_name]
        image_vector1 = model().extract_features(img_path_1)
        image_vector2 = model().extract_features(img_path_2)

        # 计算图片相似度
        out_value = cosine_similarity(image_vector1, image_vector2)
        print(f"{model_name} : {out_value}")
        output_dict[model_name] = out_value
    return output_dict


if __name__ == '__main__':
    evaluation(img_path_1="../../example/5.jpg",
               img_path_2="../../example/6.jpg",
               embedding_models=["swin_transformer", "vision_transformer", "clip"]
               )
