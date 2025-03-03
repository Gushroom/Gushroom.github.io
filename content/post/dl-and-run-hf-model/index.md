---
title: Download and run VLM model from HuggingFace
description: Some useful notes to save myself some search time
date: 2025-03-02
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

### Running Qwen-VL2.5
```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ckpt, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(ckpt)
```
What's not mentioned in the documentation, `PIL.Image` is also accepted as image input
```python
def describe_images(self, image: PIL.Image, prompt="Describe this image") -> torch.Tensor:
    """For large VLM that output natural language instead of vector embeddings"""
    from qwen_vl_utils import process_vision_info
    if self._model is None:
        self.load_model()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
                ],              
        }
    ]

    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = self.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    image_grid_thw = inputs['image_grid_thw'].numpy()
    input_h = image_grid_thw[0][1] * 14
    input_w = image_grid_thw[0][2] * 14
    inputs = inputs.to(self.device)

    generated_ids = self._model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = self.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0], (input_w, input_h)
```



