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
WIP
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
            


