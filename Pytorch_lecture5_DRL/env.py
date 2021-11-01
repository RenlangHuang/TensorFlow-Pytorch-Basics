import os
import gym
from gym.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt


register_envs = gym.envs.registry.all()
for e in register_envs: print(e)

# discrete action space
env = gym.make('CartPole-v0').unwrapped        # 使用gym库中的环境：CartPole，且打开封装
N_ACTIONS = env.action_space.n                 # 杆子动作个数 (2个)
N_STATES = env.observation_space.shape[0]      # 杆子状态维数 (4个)
print(env.action_space,env.observation_space)
# state = [x, theta, dx, dtheta]
# action = 0 or 1 (left or right)

s = env.reset()
for i in range(100):
    env.render()
    a = env.action_space.sample()
    s, r, done, _ = env.step(a)
    print(s,a,r)
    if done: break


# continuous action space
'''env = gym.make('Pendulum-v1').unwrapped
N_ACTIONS = env.action_space.shape[0]          # 杆子动作维数 (1个)
N_STATES = env.observation_space.shape[0]      # 杆子状态维数 (4个)
print(env.action_space,env.observation_space)
# state = [cos(theta),sin(theta),dtheta], -8.0<=theta dot<=8.0
# action = [joint effort(torque)], -2.0<=torque<=2.0
# reward = -(theta**2+0.1*dtheta**2+0.001*torque**2)

s = env.reset()
for i in range(100):
    env.render()
    a = env.action_space.sample()
    s, r, done, _ = env.step(a)
    print(s,a,r)
    if done: break
'''
