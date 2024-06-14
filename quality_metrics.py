import numpy as np
from skimage.metrics import structural_similarity
from PIL import Image
import imagehash


def ssim(img_1: np.ndarray, img_2: np.ndarray, max_p: int = 4095) -> float:
    """
    Structural Simularity
    """
    return structural_similarity(img_1, img_2, data_range=max_p, channel_axis=2)


def mse(img_1: np.ndarray, img_2: np.ndarray) -> float:
    """
    Mean Squared Error
    """
    return float(np.mean((img_1 - img_2) ** 2))


def phash(img_1: Image.Image, img_2: Image.Image) -> float:
    """
    Perceptual Hash + Hamming distance
    :param img_1: 第一个图像的PIL Image 对象
    :param img_2: 第二个图像的PIL Image 对象
    :return: 两个图像的感知哈希值之间的汉明距离
    """
    # 计算图片的 pHash
    phash1 = imagehash.phash(img_1)
    phash2 = imagehash.phash(img_2)

    # 计算 pHash 之间的汉明距离
    hamming_distance = phash1 - phash2

    return hamming_distance


metric_functions = {
    "ssim": ssim,
    "mse": mse,
    "phash": phash
}
