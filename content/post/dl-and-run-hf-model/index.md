---
title: Download and run SigLIP2 model from HuggingFace
description: Some useful notes to save myself some search time
date: 2025-02-22
tags: 
    - Deep Learning
    - HuggingFace
categories:
    - study notes
---
the CLIP/SigLip model family has 3 parameters. VIT-Patch-Res. VIT is the vision transformer architecture and size. Patch is when they split the input image into smaller grids, what is the size of those smaller images, 16 means 16x16, 14 means 14x14. Res is the input image resolution. The higher the more accurate but bigger(slower).
### Prereq: install git-lfs
https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md

### Download model weight:
`git clone https:/huggingface.co/<repository>`

### Running the model using transformers
```python
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# load the model and processor
ckpt = "path/to/model/directory"
model = AutoModel.from_pretrained(ckpt).to(device).eval()
processor = AutoProcessor.from_pretrained(ckpt)
tokenizer = AutoTokenizer.from_pretrained(ckpt)
inputs = processor(images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    image_embeddings = model.get_image_features(**inputs)  
```


