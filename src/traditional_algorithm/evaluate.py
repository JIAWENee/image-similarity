import cv2
from typing import List
from PIL import Image

from similarity_metrics import metric_functions


def read_image(path: str, metric: str):
    if metric in ["phash"]:
        return Image.open(path)
    else:
        return cv2.imread(path)


def _assert_image_shapes_equal(image1_path, image2_path):
    """检查图像尺寸是否一致"""
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    if image1.size != image2.size:
        raise ValueError("Input images must have the same dimensions")


def evaluation(img_path_1: str, img_path_2: str, metrics: List[str]):
    """使用传统算法评估图片相似度"""
    _assert_image_shapes_equal(img_path_1, img_path_2)
    output_dict = {}

    for metric in metrics:
        # 读取图片
        img_1 = read_image(img_path_1, metric)
        img_2 = read_image(img_path_2, metric)

        # 计算图片相似度
        metric_func = metric_functions[metric]
        out_value = float(metric_func(img_1, img_2))
        print(f"{metric.upper()} : {out_value}")
        output_dict[metric] = out_value
    return output_dict


if __name__ == '__main__':
    evaluation(img_path_1="../../example/3.jpg",
               img_path_2="../../example/3-watermark.jpg",
               metrics=["ssim", "mse", "phash"]
               )
