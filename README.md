# 图片相似度算法
计算图片相似度的应用非常广泛，典型例子包括目标跟踪、搜索引擎的以图搜图等。

相似度算法分为传统算法和深度学习算法。本项目实现了三种传统算法（SSIM、MSE、pHash）和一种深度学习算法（Embedding model + Cosine Similarity）。

# 环境配置
* Python: 3.10
* Operating system: win11

* 安装依赖：
```text
pip install -r requirements.txt
```

# 传统算法
## 算法介绍
### PHash 感知哈希
#### 算法原理
图像哈希算法通过将图像转换为固定长度的哈希值来实现图像相似度的比较和检索，主要有四种：
* 均值哈希：AHash（Average Hash）
* 感知哈希：PHash（Perceptual Hash）
* 差值哈希：DHash（Difference Hash）
* 小波哈希：WHash（Wavelet Hash）。

常用的有 AHash 和 PHash：
* 平均哈希（Average Hash）：平均哈希（AHash）算法将图像缩小到固定大小（如8x8像素），转换为灰度图像并计算其平均灰度值。然后将每个像素的灰度值与平均灰度值比较，高于平均值的像素标记为1，低于平均值的标记为0，最终组合成固定长度的哈希值。通过比较两个图像哈希值的汉明距离（Hamming Distance），可以评估图像相似度，距离越小表示相似度越高。
* 感知哈希（Perceptual Hash）： 感知哈希（PHash）算法使用离散余弦变换（DCT）提取图像频率特征。首先将图像转换为灰度图像并调整为固定尺寸（如32x32像素），然后应用DCT并保留低频分量。根据DCT系数的相对大小，将图像转换为一个二进制哈希值。通过计算两个图像哈希值的汉明距离，可以衡量图像的相似度。

#### 适用场景与优缺点
* 适用场景：简单的图像相似度比较和快速图像检索
* 优点：计算效率高、哈希值固定长度、对图像变换具有一定鲁棒性 
* 缺点：对于图像的细微变化或者复杂场景下的相似度比较可能存在一定的局限性 
* python库：https://pypi.org/project/ImageHash/


### MSE 均方误差

### SSIM 结构相似性

## 运行

# 深度学习算法
## 算法介绍
## 运行