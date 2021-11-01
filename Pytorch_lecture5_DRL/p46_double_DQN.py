import os
import gym
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


# hyper parameters
BATCH_SIZE = 32
LR = 0.01                                      # learning rate
EPSILON = 0.9                                  # greedy policy
GAMMA = 0.9                                    # reward discount
TARGET_REPLACE_ITER = 50                       # asynchronous update frequency for the target network
MEMORY_CAPACITY = 1000                         # experience pool capacity
EPOCHS = 100                                   # training epochs
env = gym.make('CartPole-v0').unwrapped        # unwrapped gym simulation CartPole
N_ACTIONS = env.action_space.n                 # discrete action space with 2 actions
N_STATES = env.observation_space.shape[0]      # continuous state
# state = [x, theta, dx, dtheta]
# action = 0 or 1 (left or right)


# Q function
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = torch.nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value


class DoubleDQN(object):
    def __init__(self, model_path, load_pretrained=True):
        self.eval_net, self.target_net = Net(), Net()
        if load_pretrained and os.path.exists(model_path):
            print('------------load the model----------------')
            self.eval_net.load_state_dict(torch.load(model_path))
        self.learn_step_counter = 0                                  # for target network update
        self.memory_counter = 0                                      # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # a transition in a line
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = torch.nn.MSELoss()

    def choose_action(self, s, greedy=False):
        s = torch.unsqueeze(torch.FloatTensor(s), 0) # 在dim=0增加维数为1的维度
        if np.random.uniform() < EPSILON or greedy:  # ε-greedy
            actions_value = self.eval_net.forward(s)
            action = torch.max(actions_value, 1)[1].data.numpy() # indices of maxQ
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        # store a transition in the experience pool
        transition = np.hstack((s, [a, r], s_))
        # cover older samples if the pool is full
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # update the target network asynchronously
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample a minbatch from the experience pool
        if self.memory_counter<MEMORY_CAPACITY:
            sample_index = np.random.choice(self.memory_counter, BATCH_SIZE)
        else: sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]) #32*4
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)) #32*1
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]) #32*1
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]) #32*4

        # evaluate each action with eval_net, gather Q values from each line according to indices b_a
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # no backpropagation(detach)
        q_next = self.target_net(b_s_).detach()
        # choose optimal actions by eval_net
        a_next = self.eval_net(b_s_).detach().argmax(1).reshape(-1,1)
        # evaluate the target Q-value
        q_next = q_next.gather(1,a_next)
        # according to Bellman equation
        q_target = b_r + GAMMA * q_next.reshape(-1,1)
        
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data.item()


ddqn = DoubleDQN('./checkpoint/double_dqn.pth')
losses, rewards = [], []
for i in range(EPOCHS):
    s = env.reset()
    episode_reward_sum = 0

    while True:
        env.render()
        a = ddqn.choose_action(s)
        s_, r, done, info = env.step(a)

        # well defined reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        new_r = r1 + r2

        ddqn.store_transition(s, a, new_r, s_)
        episode_reward_sum += new_r

        s = s_

        if ddqn.memory_counter > BATCH_SIZE:
            losses.append(ddqn.learn()) # experience replay

        if done:
            print('episode %d, reward_sum: %.2f' % (i, episode_reward_sum))
            rewards.append(episode_reward_sum)
            break

torch.save(ddqn.eval_net.state_dict(),'./checkpoint/double_dqn.pth')
plt.subplot(1,2,1)
plt.title('training loss')
plt.plot(losses)
plt.grid()
plt.subplot(1,2,2)
plt.title('model evaluation')
plt.plot(rewards,label='long-term reward')
plt.legend()
plt.grid()
plt.show()


# application
s = env.reset()
while True:
    env.render()
    with torch.no_grad():
        a = ddqn.choose_action(s,True)
    s, _, done, _ = env.step(a)
    if done: break
