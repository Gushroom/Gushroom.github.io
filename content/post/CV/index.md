---
title: Computer Vision
description: Key understandings and notes of computer vision
math: true
date: 2025-02-25
tags: 
    - Deep Learning
    - Computer Vision
categories:
    - study notes
---
## CNN
### Convolution
卷积 = 特征提取
- 索伯算子 https://en.wikipedia.org/wiki/Sobel_operator
    - 既是传统的图像处理，也是很好的特征提取例子
为什么卷积的同时通道数也在增加？
- 卷积和池化都会减少dimension，增加通道数来保证不丢失太多信息量
- `n_channels`个卷积核随机初始化，对同一个feature map进行特征提取
    - 每个通道可以学习不同的特征
- 1x1卷积核可以在不改变feature map大小的前提下改变通道数
    - 升维/降维
    - 通道间传递信息：使用1x1卷积核，实现降维和升维的操作其实就是channel间信息的线性组合变化，3x3，64channels的卷积核后面添加一个1x1，28channels的卷积核，就变成了3x3，28channels的卷积核，原来的64个channels就可以理解为跨通道线性组合变成了28channels，这就是通道间的信息交互
### Key components of CNN:
- Convolution: Extract feature from images, value in each cell are randomized and learned.
- Pooling: Reduce dimensionality to prevent overfitting, reduce computational cost(reduce size of feature map).
- Activation: Introduce Non-linearity. 
- Fully connected layers: Perform classification.


### Size calculation: 
- Convolution: $\tt{size} = \lfloor \frac{(\tt{width - kernelsize} + 2 \times \tt{padding})}{\tt{stride}} \rfloor + 1$
- Pooling: $\tt{size} = \lceil \frac{size}{2} \rceil$
- When size remains the same: kernelsize = 3 and padding = 1, kernel = 5 and padding = 2
## VIT (Vision Transformer)
- Trains faster than CNN (transformer is more suitable for parallel computation)
- Requires slightly more dataset to train (at smaller datasets lose to CNN, excels at larger datasets)
- Attempt to apply transformer achitecture on image tasks without change
    - represent image features like text
        - patch(224 = 14px * 14px * (4x4)grid)
            1. Flattern pixels in the patch
            2. Linear projection of flatterned patches
            3. Add a learnable position encoder, attach to image token vectors
            


