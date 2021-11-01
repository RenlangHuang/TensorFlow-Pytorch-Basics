import os
import gym
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 超参数
BATCH_SIZE = 32
LR = 0.01                                      # learning rate
EPSILON = 0.9                                  # greedy policy
GAMMA = 0.9                                    # reward discount
TARGET_REPLACE_ITER = 50                       # 目标网络更新频率
MEMORY_CAPACITY = 1000                         # 经验池容量
EPOCHS = 150                                   # training epochs
env = gym.make('Pendulum-v1').unwrapped        # 使用gym库中的环境：CartPole，且打开封装
N_ACTIONS = env.action_space.shape[0]          # 杆子动作个数 (2个)
N_STATES = env.observation_space.shape[0]      # 杆子状态个数 (4个)
print(env.action_space,env.observation_space)
# state = [cos(theta),sin(theta),dtheta], -8.0<=theta dot<=8.0
# action = [joint effort(torque)], -2.0<=torque<=2.0
# reward = -(theta**2+0.1*dtheta**2+0.001*torque**2)

s = env.reset()
while True:
    env.render()
    a = env.action_space.sample()
    s, r, done, _ = env.step(a)
    print(s,a,r)
    if done: break
