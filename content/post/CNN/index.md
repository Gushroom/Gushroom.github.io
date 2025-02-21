---
title: CNN
description: Key understandings and notes of CNN
math: true
date: 2025-02-20
tags: 
    - Deep Learning
    - Computer Vision
categories:
    - study notes
---
## Key components of CNN:
- Convolution: Extract feature from images, value in each cell are randomized and learned.
- Pooling: Reduce dimensionality to prevent overfitting, reduce computational cost(reduce size of feature map).
- Activation: Introduce Non-linearity. 
- Fully connected layers: Perform classification.

Size calculation: 
- Convolution: $\tt{size} = \lfloor \frac{(\tt{width - kernelsize} + 2 \times \tt{padding})}{\tt{stride}} \rfloor + 1$
- Pooling: $\tt{size} = \lceil \frac{size}{2} \rceil$
- When size remains the same: kernelsize = 3 and padding = 1, kernel = 5 and padding = 2
