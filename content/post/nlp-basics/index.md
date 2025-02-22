---
title: Overview of NLP(LLM)
description: Diving deeper into NLP(LLM) basics, from encoder to RLHF.
math: true
date: 2025-02-21
tags: 
    - Deep Learning
    - LLM
    - NLP
categories:
    - study notes
---

## What is encoder and decoder?
### Encoding
Encoding is the process of extracting semantic information from languge. For some representation to be a good choice of encoding used in machine learning, it must satisfy 2 standards: 
1. It must be easily digitized(understood by computers)
2. The relationship between encodings must somehow represent the relationship between original language tokens.
Let's focus on (1.) first. To that end, we have tokenizer and one-hot encoding.
- Tokenizer gives each token an unique id, thus projecting the entire token space onto a one-dimensional axis(array).
- One-hot encoding uses unique binary ids, projecting each token onto it's own dimension, keeping distances between encoded tokens the same.

However they both fail to achieve the second standard. 
- Tokenizer makes it simple to calculate relationships(difference) between encoded tokens, but fails to capture the complex relationships between tokens, especially when some words have multiple meanings in different contexts.
- One-hot encoding cannot represent the relationship between tokens with relative distance at all. Token vectors are all perpendicular to each other, and the dot product is always 0. However, it makes addition simple, we can have combination of any vectors.
### Latent space and embedding
We need some intermediate space, uses advantages from both tokenizer that captures relationship but is too dense, and one-hot encoding that allows easy combination(addition) of vectors but is too sparse. We can either add more dimensions to the one-dimensional tokenizer, or reduce dimensions from one-hot encoded space. Embedding is the process of reducing dimensionality from one-hot encoding by neural networks to a lower dimensional latent space.
### How do we poject token into embedding?
We need to find a relationship to properly embed tokens as vectors.
#### Word2Vec
In 2013, Google proposed Word2Vec model, the first low-dimension word embedding, introducing two breakthroughs: Continuous Bag of Words(CBOW) and skip-gram. 
- CBOW: collects words that appear before and after a target word in a sequence. For example, "The cat sits on the mat." and the target word is "sits", the model learns by trying to predict "sits" from ["The", "cat", "on", "the"]. The model learns a weight to transform words into embeddings that has the max probability to predict the right word.
- Skip-grams: the opposite, predicts n context words based on a given target word.

Problems with Word2Vec:
- It only considers one meaning of each word. (i.e. bank)
- Most importantly, it does not consider long range context information.
### LSTM
Better aptures long range context, but still has a lot of issues:
- LSTM process word one by one, this cannot be parallelized, making training slow.
- Still struggles with longer context length.
WIP
### Transformers
#### Self Attention
$$\tt{Attention}(\mathbf{Q, K, V}) = \tt{softmax} \lparen \frac{\mathbf{Q}^T\mathbf{K}}{\sqrt{d_k}} \rparen \mathbf{V}$$
An interesting observation is $\mathbf{Q^TK}$ is matrix multiplication, but it is called "dot product" because it effectively computes row-wise similarity score, and dot product is usually used for that purpose.
```python
class SelfAttention(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        # Transform input X into embedding space
        # X shape: (batch_size, sequence_length, embedding_size)
        Q = self.query(x) # shape: (batch_size, sequence_length, embedding_size)
        K = self.key(x)
        V = self.value(x)

        d_k = self.embedding_size

        score = torch.matmul(Q, K.transpose(-2, -1)) # this is the Query-wise similarity score
        # We transpose K here because Q and K have shape = (B, S, E), and K.transpose(-2, -1) gives us (B, E, S)
        # Q @ K.T then gives us (S, S) for each batch, which is the similarity between each QK-pair

        normalized_score = score / torch.sqrt(d_k)
        # score grows as d increases, because it is computed as sum of products
        # Dividing by d_k is too big, shrinking the magnitude too much.

        attention_weights = nn.functional.softmax(normalized_score, dim = -1)

        output = torch.matmul(attention_weights, V)
        return output
```
A `Linear()` layer computes $y = xW^T + b$, it tranforms input vector $x$ into another space by weight matrix $W$ and bias $b$. 
In this case, it projects $x$ onto the embedding space.

### Multihead Attention
WIP