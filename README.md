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

# RUN
## 传统算法
进入`src/traditional_algorithm`, 在`evaluate.py`中指定要比较的图片路径，执行：
```
python evaluate.py
```
## 深度学习算法
1. 下载模型, 在`src/deep_learning/embedding_model.py`中配置模型路径。模型下载地址：
* [swin_transformer](https://huggingface.co/microsoft/swin-base-patch4-window7-224)
* [vision_transformer](https://huggingface.co/google/vit-base-patch16-224-in21k)
* [clip](https://huggingface.co/openai/clip-vit-base-patch32)
2. 进入`src/deep_learning`, 在`evaluate.py`中指定要比较的图片路径和要使用的算法模型，执行：
```
python evaluate.py
```


# 算法介绍
## 传统算法
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
* python库：[imagehash](https://pypi.org/project/ImageHash/)


### MSE 均方误差
均方误差（Mean Squared Error, MSE）是一种常用的图片相似度算法，用于衡量两张图片之间的差异程度。
* 计算方法：通过计算两张图片对应像素之间的差值的平方，并求取平均值来得到相似度评分。
* 缺点：只考虑像素级别的差异，可能无法准确地捕捉图像的纹理、结构等细节。
* 取值范围：值越小表示两张图片越相似，值为0表示完全相同。

### SSIM 结构相似性
结构相似性（Structural Similarity, SSIM）算法综合考虑了图像在亮度、对比度和结构三个方面的差异.
* 优点： 与均方误差（MSE）相比，SSIM更能捕捉图像的结构信息和感知差异。
* 适用场景： 
  1. 用于衡量两张图片之间结构相似性 
  2. 对于比较不同字体的字形相似度，可以考虑使用SSIM算法。因为SSIM算法更注重图像的结构相似性，它更好地检测出字形上的细微差异。
* 取值范围：范围[0, 1]，值越大，表示图像失真越小
* python库：[structural_similarity](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity)

## 深度学习算法
通过深度学习模型，将图片转化为向量表示，并使用相似度度量方法（如欧氏距离、余弦相似度、汉明距离等）计算相似度得分。在本项目中，使用了以下三种深度学习模型将图片转为向量表示，并通过余弦相似度计算图片的相似度。

[以下部分数据来源GPT，仅供参考]

|   模型名称   |                                                       vision transformer                                                        |                                                               swin transformer                                                                |                                                                           CLIP                                                                            |
|:--------:|:-------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  模型开源地址  |                           [vision_transformer](https://github.com/google-research/vision_transformer)                           |                             [swin transformer](https://github.com/microsoft/Swin-Transformer?tab=readme-ov-file)                              |                                              [CLIP](https://github.com/openai/CLIP/blob/main/model-card.md)                                               |
|  模型下载地址  |                    [vision_transformer](https://huggingface.co/google/vit-base-patch16-224-in21k/tree/main)                     |                          [swin transformer](https://huggingface.co/microsoft/swin-base-patch4-window7-224/tree/main)                          |                                           [CLIP](https://huggingface.co/openai/clip-vit-base-patch32/tree/main)                                           |
|   训练数据   |                                                      ImageNet、COCO、ADE20K等                                                      |                                                            ImageNet-21k、JFT-300M等                                                             |                                                                          图像-文本对                                                                           |
|   嵌入长度   |                                                               768                                                               |                                                                     1024                                                                      |                                                                     图像: 512, 文本: 512                                                                      |
|   模型大小   |                                                             2.57 GB                                                             |                                                                    1.96 GB                                                                    |                                                                          3.38 GB                                                                          |
|   通用性    | ViT 是一种基于自注意力机制的模型，它将图像划分为一系列小块（patch），然后将这些小块序列化并输入到 Transformer 中进行处理。由于 Transformer 架构的高度通用性，ViT 可以用于各种视觉任务，如图像分类、对象检测和图像分割。 | Swin Transformer 是对 ViT 的一种改进，采用了层级化的结构和移动窗口（shifted window）机制，能够更好地捕捉图像中的局部和全局信息。Swin Transformer 同样适用于多种视觉任务，包括分类、检测和分割。它在处理高分辨率图像时，表现尤为优异。 | CLIP（Contrastive Language–Image Pretraining）是 OpenAI 提出的模型，旨在将视觉和语言信息融合在一起。CLIP 能够通过联合训练图像和文本来实现跨模态任务，例如图像检索、图像描述和零样本分类。CLIP 的多模态设计使得它在处理涉及图像和文本的任务时非常通用。 |
|   准确性    |                                         ViT在大规模图像分类任务上表现良好，但在细粒度任务上可能不如Swin Transformer                                         |                                                    在多个视觉任务（如图像分类、物体检测和语义分割）上表现出色，通常比ViT更准确                                                    |                                                          结合图像和文本的对比学习，能有效提取跨模态特征，在开放域图像识别任务重表现出色                                                          |
|   鲁棒性    |                                           由于其全局注意机制，可能在处理图像局部变化时不如Swin Transformer稳健                                            |                                                  通过分层结构和局部注意机制（移动窗口机制），对光照变化、噪声和角度变化有较好的鲁棒性                                                   |                                                     在处理多样化和未见过的图像和文本组合时表现出色，但在细粒度的视觉任务上可能不如专门设计的视觉模型                                                      |
|   计算效率   |                                                  ViT的全局注意机制在处理高分辨率图像时计算复杂度较高。                                                   |                                            Swin Transformer 的分层结构和移动窗口机制减少了计算复杂度，使其在处理高分辨率图像时更加高效。                                            |                                                                 训练和推理时涉及图像和文本的处理，计算复杂度较高。                                                                 |
|   资源消耗   |                                                     ViT在处理高分辨率图像时，GPU内存消耗较高                                                     |                                                   Swin Transformer 的设计使其在相同条件下比ViT消耗更少的计算资源                                                   |                                                                    需要同时处理图像和文本，资源需求较大                                                                     |
