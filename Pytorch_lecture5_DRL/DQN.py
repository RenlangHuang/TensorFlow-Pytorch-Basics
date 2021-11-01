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
EPOCHS = 200                                   # training epochs
env = gym.make('CartPole-v0').unwrapped        # 使用gym库中的环境：CartPole，且打开封装
N_ACTIONS = env.action_space.n                 # 杆子动作个数 (2个)
N_STATES = env.observation_space.shape[0]      # 杆子状态个数 (4个)


# 定义Net类 (定义Q函数网络)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)        # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.out = torch.nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)        # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value


# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self, model_path, load_pretrained=True):                       # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(), Net()                           # 利用Net创建两个神经网络: 评估网络和目标网络
        if load_pretrained and os.path.exists(model_path):
            print('------------load the model----------------')
            self.eval_net.load_state_dict(torch.load(model_path))
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))             # 初始化经验池，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = torch.nn.MSELoss()

    def choose_action(self, s, greedy=False):                                   # 定义动作选择函数
        s = torch.unsqueeze(torch.FloatTensor(s), 0)                            # 将s转换成32-bit float形式，并在dim=0增加维数为1的维度
        if np.random.uniform() < EPSILON or greedy:                             # ε-greedy
            actions_value = self.eval_net.forward(s)
            action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy形式
            action = action[0]                                                  # 输出action的第一个数
        else:                                                                   # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))                                 # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY                           # 获取transition要置入的行数
        self.memory[index, :] = transition                                      # 置入transition
        self.memory_counter += 1                                                # memory_counter自加1

    def learn(self):
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1                                            # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]                                 # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data.item()


dqn = DQN('./checkpoint/dqn.pth')
'''losses, rewards = [], []
for i in range(EPOCHS):
    s = env.reset()                        # 重置环境
    episode_reward_sum = 0                 # 初始化该循环对应的episode的总奖励

    while True:                            # 开始一个episode (每一个循环代表一步)
        env.render()                       # 显示实验动画
        a = dqn.choose_action(s)           # 输入该步对应的状态s，选择动作
        s_, r, done, info = env.step(a)    # 执行动作，获得反馈

        # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        new_r = r1 + r2

        dqn.store_transition(s, a, new_r, s_)
        episode_reward_sum += new_r        # 逐步加上一个episode内每个step的reward

        s = s_                             # 更新状态

        if dqn.memory_counter > MEMORY_CAPACITY:
            losses.append(dqn.learn()) # experience replay

        if done:       # 如果done为True
            print('episode %d, reward_sum: %.2f' % (i, episode_reward_sum))
            rewards.append(episode_reward_sum)
            break      # 该episode结束

torch.save(dqn.eval_net.state_dict(),'./checkpoint/dqn.pth')
plt.subplot(1,2,1)
plt.title('training loss')
plt.plot(losses)
plt.grid()
plt.subplot(1,2,2)
plt.title('model evaluation')
plt.plot(rewards,label='long-term reward')
plt.legend()
plt.grid()
plt.show()'''


# application
s = env.reset()
while True:
    env.render()
    with torch.no_grad():
        a = dqn.choose_action(s,True)
    s, _, done, _ = env.step(a)
    if done: break