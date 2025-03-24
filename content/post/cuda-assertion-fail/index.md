---
title: Cuda Assertion Fail
description: 
math: true
date: 2025-03-24
tags: 
    - Deep Learning
categories:
    - misc
---

```
/opt/conda/conda-bld/pytorch_1699449181202/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [25778,0,0], thread: [37,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1699449181202/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [25778,0,0], thread: [38,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1699449181202/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [25778,0,0], thread: [39,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1699449181202/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [25778,0,0], thread: [40,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1699449181202/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [25778,0,0], thread: [41,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1699449181202/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [25778,0,0], thread: [42,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
```

Try catch doesn't work here. If anything we should try to `assert` before crashing out like that

Moving to CPU will help debugging.
