import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

# Implementation of Deep Deterministic Policy Gradient (DDPG)
# Paper: Continuous Control with Deep Reinforcement Learning, https://arxiv.org/abs/1509.02971

# hyper parameters
MAX_EPISODES = 100
MAX_EP_STEPS = 200
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9   # reward discount
TAU = 0.01    # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

env = gym.make('Pendulum-v1').unwrapped
env.seed(1) # facilitate the repetition
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
print(env.action_space,env.observation_space)
# state = [cos(theta),sin(theta),dtheta], -8.0<=theta dot<=8.0
# action = [joint effort(torque)], -2.0<=torque<=2.0
# reward = -(theta**2+0.1*dtheta**2+0.001*torque**2)


class ActorNet(torch.nn.Module):
    def __init__(self,s_dim,a_dim):
        super(ActorNet,self).__init__()
        self.fc1 = torch.nn.Linear(s_dim,30)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = torch.nn.Linear(30,a_dim)
        self.out.weight.data.normal_(0,0.1)
    
    def forward(self,x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions_value = x*2
        return actions_value

class CriticNet(torch.nn.Module):
    def __init__(self,s_dim,a_dim):
        super(CriticNet,self).__init__()
        self.fcs = torch.nn.Linear(s_dim,30)
        self.fcs.weight.data.normal_(0,0.1)
        self.fca = torch.nn.Linear(a_dim,30)
        self.fca.weight.data.normal_(0,0.1)
        self.out = torch.nn.Linear(30,1)
        self.out.weight.data.normal_(0,0.1)
    
    def forward(self,s,a):
        x = self.fcs(s)
        y = self.fca(a)
        net = torch.relu(x+y)
        actions_value = self.out(net)
        return actions_value


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, model_path, load_pretrained=True):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim*2 + a_dim + 1), dtype=np.float32)
        self.memory_counter = 0
        self.actor_eval = ActorNet(s_dim,a_dim)
        self.actor_target = ActorNet(s_dim,a_dim)
        self.critic_eval = CriticNet(s_dim,a_dim)
        self.critic_target = CriticNet(s_dim,a_dim)

        if load_pretrained and os.path.exists(model_path[0]):
            print('------------load the model----------------')
            self.actor_eval.load_state_dict(torch.load(model_path[0]))
            self.actor_target.load_state_dict(torch.load(model_path[0]))
            self.critic_eval.load_state_dict(torch.load(model_path[1]))
            self.critic_target.load_state_dict(torch.load(model_path[1]))

        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(),lr=LR_C)
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(),lr=LR_A)
        self.loss_func = torch.nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.actor_eval(s)[0].detach()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % MEMORY_CAPACITY # replace the old memory
        self.memory[index, :] = transition
        self.memory_counter += 1

    def soft_update(self,target,source,epsilon=0.1):
        for target_param, source_param in zip(target.parameters(),source.parameters()):
            target_param.data.copy_((1-epsilon)*target_param.data + epsilon*source_param.data)

    def train(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        # critic: gradient descent
        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_,a_)
        q_target = br + GAMMA * q_
        q_eval = self.critic_eval(bs,ba)
        td_error = self.loss_func(q_target,q_eval)
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

        # actor: gradient ascent
        a = self.actor_eval(bs)
        q = self.critic_eval(bs,a)
        loss_a = -torch.mean(q)
        self.actor_optimizer.zero_grad()
        loss_a.backward()
        self.actor_optimizer.step()

        # soft target replacement
        self.soft_update(self.actor_target,self.actor_eval,TAU)
        self.soft_update(self.critic_target,self.critic_eval,TAU)
        return [loss_a.data.item(),td_error.data.item()]


model_path = ('./checkpoint/ddpg_actor.pth','./checkpoint/ddpg_critic.pth')
ddpg = DDPG(a_dim, s_dim, a_bound, model_path)

var = 3  # control exploration
losses, rewards = [], []
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_rs_sum = 0
    for j in range(MAX_EP_STEPS):
        env.render()
        # add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)
        s_, r, _, _ = env.step(a)

        ddpg.store_transition(s, a, r/10, s_)

        s = s_

        if ddpg.memory_counter > MEMORY_CAPACITY:
            if var>0.03: var *= 0.9995 # decay the action randomness
            losses.append(ddpg.train())
        ep_rs_sum += r

        if 'episode_reward_sum' not in globals():
            episode_reward_sum = ep_rs_sum
        else:
            episode_reward_sum = episode_reward_sum * 0.99 + ep_rs_sum * 0.01

    print('episode %d, reward_sum: %.2f, explore: %.2f' % (i, episode_reward_sum, var))
    rewards.append(episode_reward_sum)

torch.save(ddpg.actor_eval.state_dict(),model_path[0])
torch.save(ddpg.critic_eval.state_dict(),model_path[1])
losses = np.array(losses)
plt.subplot(1,2,1)
plt.title('training loss')
plt.plot(losses[:,0],label='actor error')
plt.plot(losses[:,1],label='critic error')
plt.legend()
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
        a = ddpg.choose_action(s)
    s, _, done, _ = env.step(a)
    if done: break