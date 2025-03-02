---
title: Overview of NLP(LLM)
description: Diving deeper into NLP(LLM) basics, from encoder to RLHF.
math: true
date: 2025-02-25
tags: 
    - Deep Learning
    - LLM
    - NLP
categories:
    - study notes
---
Answering all of my own questions about what I know that I dont know.
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
### Tokenizer - Splitting text corpuses into language primitives
#### BPE (Byte-Pair Encoding)
A very detailed step by step explanation can be found here: https://huggingface.co/learn/nlp-course/en/chapter6/5

Split text corpuses into primitives, such as individual characters(ascii, unicode), then iteratively merge the most frequent combination of the current primitives into a new token. We merge until we have got a satisfactory vocabulary size. 
##### BBPE 
Ascii and unicode have problems with representing unknown chars, such as Chinese, emoji. BBPE splits corpuses into bytes, having better representation and support for mulitlingual and special characters.

### How do we poject token into embedding?
We need to find a relationship to properly embed tokens as vectors.
#### Word2Vec
In 2013, Google proposed Word2Vec model, the first low-dimension word embedding, introducing two breakthroughs: Continuous Bag of Words(CBOW) and skip-gram. 
- CBOW: collects words that appear before and after a target word in a sequence. For example, "The cat sits on the mat." and the target word is "sits", the model learns by trying to predict "sits" from ["The", "cat", "on", "the"]. The model learns a weight to transform words into embeddings that has the max probability to predict the right word.
- Skip-grams: the opposite, predicts n context words based on a given target word.
##### What it does:
- King - man + women $\approx$ queen. Vector carries semantic meanings and can be "calculated".
- Cosine similarity can actually find similarity between semantically close vectors.
##### Problems with Word2Vec:
- It only considers one meaning of each word. (i.e. bank)
- Most importantly, it does not consider long range context information.

### LSTM
Better aptures long range context, but still has a lot of issues:
- LSTM process word one by one, this cannot be parallelized, making training slow.
- Still struggles with longer context length.

WIP...Not tooo interested in RNN and LSTM and GRU tho...
### Transformers
#### Self Attention
$$\tt{Attention}(\mathbf{Q, K, V}) = \tt{softmax} \lparen \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \rparen \mathbf{V}$$
An interesting observation is $\mathbf{QK}^T$ is matrix multiplication, but it is called "dot product" because it effectively computes row-wise similarity score, and dot product is usually used for that purpose.

This similarity computation "assigns more weight" to any previous token $K$ that has a closer relationship with $Q$, allows the model to focus on more relevant information.
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

The `softmax()` function computes $\tt{softmax}(x_i) = \frac{e^{x_i}}{\sum^n_{j=1}e^{x_j}}$. It projects points on the real number axis $x_i$ onto the function $y_i = e^{x_i}$. Then it considers the sum of all $y$ values to be 1. Now we have a non-zero probability score.

Another interesting fact about the softmax function is: if we multiply every $x$ with a factor, the relationship between $y$ will change. When this factor is less than 1, we will see the probability distribution moving towards a uniform distribution. This is the **temperature** parameter. The lower the temperature, the more "random" the results are.
#### What's wrong with self attention?
It only has one QKV set. The model only learns one set of weight matrix, limiting its ability to capture more features of the data.
### Multihead Attention
Proposed in the **Attention is All You Need** paper in 2017 with the transformers achitecture.
```python
class MultiheadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        assert embedding_size % num_heads == 0, "Embedding size must be divisible by num_heads"

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)

        self.out = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        batch_size, sequence_length, embedding_size = x.shape

        Q = self.query(x) # shape: (batch_size, sequence_length, embedding_size)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        # Q,K,V.shape(batch_size, self.num_heads, sequence_length, self.head_dim)
        # Why view + transpose instead of view(batch_size, sequence_length, self.num_heads, self.head_dim)?
        #   This is because how view() split tensor dimensions, we want to split embedding_size to num_heads * head_dim
        #   But in the end we want (batch_size, self.num_heads, sequence_length, self.head_dim) for future calculations

        scores = torch.matmul(Q, K.transpose(-2, -1))
        # K.T.shape: (batch_size, self.num_heads, self.head_dim, sequence_length)
        # Q @ K.T shape: (batch_size, self.num_heads, sequence_length, sequence_length)
        # (sequence_length, sequence_length) representing the attention scores for each head across all sequence positions
        normalized_scores = scores / self.head_dim ** 0.5

        attention_weights = nn.functional.softmax(normalized_scores, dim = -1)
        attention_outputs = torch.matmul(attention_weights, V)
        # V.shape: (batch_size, self.num_heads, sequence_length, self.head_dim)
        # attention_output shape: (batch_size, self.num_heads, sequence_length, head_dim)

        combined_heads = attention_outputs.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)
        # attention_outputs.transpose(1, 2): (batch_size, sequence_length, self.num_heads, head_dim)

        output = self.out(combined_heads)
        return output
``` 
Note that besides the split and join of multiple attention heads, in the final layer we project attention output into embedding space. 
#### What's wrong with MHA?
MHA is great in terms of quality, the problem is computation cost. In MHA we repeat the attention calculation `num_heads` times, and that starts to become a problem when the model size get bigger. In order to compute `attention_outputs`, we need to load the QKV matricies into memory many times. and data transferring also becomes a bottleneck.

So we reduce amount of heads.

![Diagram of MHA, MQA, and GQA](MHA-MQA-GQA.png)
### Multi Query Attention
Here we keep only one K and V, and we split Q into `num_heads`. 
```python
class MultiQueryAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        assert embedding_size % num_heads == 0, "Embedding size must be divisible by num_heads"
        
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        
        # Separate query projection for multiple heads
        self.query = nn.Linear(embedding_size, embedding_size)
        # Shared key and value projection (single head)
        self.key = nn.Linear(embedding_size, self.head_dim)
        self.value = nn.Linear(embedding_size, self.head_dim)
        
        self.out = nn.Linear(embedding_size, embedding_size)
        
    def forward(self, x):
        batch_size, sequence_length, embedding_size = x.shape
        
        Q = self.query(x)  # (batch_size, sequence_length, embedding_size)
        K = self.key(x)    # (batch_size, sequence_length, head_dim)
        V = self.value(x)  # (batch_size, sequence_length, head_dim)
        
        # Reshape queries to multiple heads
        Q = Q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        # Keys and values are shared across all heads, so they have a single head dimension
        K = K.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        V = V.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  
        # K.unsqueeze(1) = (batch_size, 1, sequence_length, head_dim)
        # K.expand = (batch_size, num_heads, sequence_length, head_dim)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5
        # Q @ K.T = (batch_size, num_heads, sequence_length, sequence_length)
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attention_outputs = torch.matmul(attention_weights, V)
        
        combined_heads = attention_outputs.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)
        output = self.out(combined_heads)
        return output
```
### Group Query Attention
Between MLA and MQA, uses G(1 << G << num_heads) Groups of KV pair, and still num_heads amount of Q matrices.

When G = 1, GQA = MQA, when G = num_heads, GQA = MHA.
```python
class GroupQueryAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, num_groups):
        super().__init__()
        assert embedding_size % num_heads == 0, "embedding_size must be divisible by num_heads!"
        assert embedding_size % num_groups == 0, "embedding_size must be divisible by num_groups!"
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups!"
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embedding_size // num_heads 
        self.heads_per_group = num_heads // num_groups

        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size // self.heads_per_group) 
        self.value = nn.Linear(embedding_size, embedding_size // self.heads_per_group)

        self.out = nn.Linear(embedding_size, embedding_size)

    
    def forward(self, x):

        batch_size, sequence_length, embedding_size = x.shape

        Q = self.query(x) # (batch_size, sequence_length, embedding_size)
        K = self.key(x) # (batch_size, sequence_length, embedding_size // heads_per_group)
        V = self.value(x) # (batch_size, sequence_length, embedding_size // heads_per_group)

        Q = Q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.num_groups, -1).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.num_groups, -1).transpose(1, 2)
        # (batch_size, num_groups, sequence_length, head_dim)

        K = K.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        V = V.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        # expand heads_per_group into the third dimension
        # (batch_size, num_groups, heads_per_group, sequence_length, head_dim)

        K = K.contiguous().view(batch_size, self.num_heads, sequence_length, self.head_dim)

```

