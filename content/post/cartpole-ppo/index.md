---
title: Cartpole
description: Getting started with RL
date: 2025-03-02
math: true
tags: 
    - Deep Learning
    - Reinforcement Learning
categories:
    - study notes
---

## Policy Gradient
强化学习的核心之一就是**学以致用**。一边在探索环境中学习，一边用学到的action来获取reward。Policy是环境到行为的映射，policy(state) = action, 同理还有state(action) = reward。

At every step, the agent takes an action based on the policy. If the action space is discrete, the policy should return a softmax probability of each action, and if the action space is continuous, the policy outputs the mean and standard deviation of the Gaussian probability distribution for each continuous action.

为什么连续的空间要输出高斯分布呢？
- 均值（$\mu$）：代表智能体认为在当前状态下最优的动作值。
- 标准差（$\theta$）：控制探索的随机性（$\theta$ 越大，动作的随机性越强）。

Policy 既然是一个映射（函数），那我们就可以用神经网络来拟合

CartPole的环境比较简单，用一个小MLP就可以了

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```
每一步计算长期return
Return是从这一步加后续所有奖励折扣之和
discount_factor一般小于1，越远的回报权重越小
在最前面插入，确保顺序是`[R0, R1, R2...]`
```python
def calculate_stepwise_returns(rewards, discount_factor):
    returns = []
    R = 0
    for r in reversed(rewards): # 先
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    normalized_returns = (returns - returns.mean()) / returns.std()
    return normalized_returns
```
为什么要Normalize?
- 减小方差
- 加速收敛
```python
def forward_pass(env, policy, discount_factor):
    log_prob_actions = []
    rewards = []
    done = False
    episode_return = 0
    policy.train()
    observation, info = env.reset()
    while not done:
        observation = torch.FloatTensor(observation).unsqueeze(0) # state
        action_pred = policy(observation) # action = policy(state)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob) # discretize
        action = dist.sample() 
        log_prob_action = dist.log_prob(action)
        observation, reward, terminated, truncated, info = env.step(action.item()) # reward = state(action)
        done = terminated or truncated
        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        episode_return += reward
    log_prob_actions = torch.cat(log_prob_actions)
    stepwise_returns = calculate_stepwise_returns(rewards, discount_factor)
    return episode_return, stepwise_returns, log_prob_actions
```
计算loss
```python
def calculate_loss(stepwise_returns, log_prob_actions):
    loss = -(stepwise_returns * log_prob_actions).sum()
    return loss
```
这里的loss function就是最基本的policy gradient, REINFORCE

$$ \tt{Loss} = -\sum_t (\tt{Return}_t \times \log \pi (a_t | s_t))$$

$\log \pi (a_t | s_t)$ 是策略网络输出的动作对数概率（`log_prob_actions`）

并反向传播更新权重
```python
def update_policy(stepwise_returns, log_prob_actions, optimizer):
    stepwise_returns = stepwise_returns.detach()
    loss = calculate_loss(stepwise_returns, log_prob_actions)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```
训练并保存权重
```python
import gymnasium as gym
import torch
import numpy as np
from torch import optim

env = gym.make("CartPole-v1")
seed = 66
np.random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)

def main(): 
    MAX_EPOCHS = 500
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    INPUT_DIM = env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = env.action_space.n
    DROPOUT = 0.5
    episode_returns = []
    policy = PolicyNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
    LEARNING_RATE = 0.01
    optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)
    for episode in range(1, MAX_EPOCHS+1):
        episode_return, stepwise_returns, log_prob_actions = forward_pass(env, policy, DISCOUNT_FACTOR)
        _ = update_policy(stepwise_returns, log_prob_actions, optimizer)
        episode_returns.append(episode_return)
        mean_episode_return = np.mean(episode_returns[-N_TRIALS:])
        if episode % PRINT_INTERVAL == 0:
            print(f'| Episode: {episode:3} | Mean Rewards: {mean_episode_return:5.1f} |')
        if mean_episode_return >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            torch.save(policy, 'pg.pt')
            break

if __name__ == "__main__":
    main()
```
Load model and let it play
```python
import gymnasium as gym
import torch

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
policy = torch.load("ppo.pt", weights_only=False).eval()

episode_over = False
while not episode_over:
    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    
    # Get action from policy
    with torch.no_grad():
        action_logits = policy(obs_tensor)
        action = torch.argmax(action_logits, dim=1).item()
        
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()
```









