import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

# Implementation of Advantage Actor Critic (A2C), no papers

class ActorCriticNet(torch.nn.Module):
    def __init__(self,s_dim,a_dim):
        super(ActorCriticNet, self).__init__()
        self.fcl = torch.nn.Linear(s_dim, 128)
        self.actor = torch.nn.Linear(128, a_dim)
        self.critic = torch.nn.Linear(128, 1)
    
    def forward(self, x):
        hidden = torch.relu(self.fcl(x))
        action = torch.softmax(self.actor(hidden),dim=-1)
        value = self.critic(hidden)
        return action, value


class A2C(object):
    def __init__(self, a_dim, s_dim, model_path, load_pretrained=True):
        self.a_dim, self.s_dim = a_dim, s_dim
        self.actor_critic = ActorCriticNet(s_dim,a_dim)
        
        if load_pretrained and os.path.exists(model_path):
            print('------------load the model----------------')
            self.actor_critic.load_state_dict(torch.load(model_path))
        
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(),lr=LR)
        self.loss_func = torch.nn.SmoothL1Loss()
        self.state_values, self.rewards, self.loglikelihood = [], [], []

    def choose_action(self, s):
        self.actor_critic.eval()
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        action, value = self.actor_critic(s)
        actions = torch.distributions.Categorical(action[0])
        action = actions.sample()
        loglikelihood = actions.log_prob(action)
        
        self.loglikelihood.append(loglikelihood)
        self.state_values.append(value[0][0])
        return action.data.item()

    def store_transition(self, r):
        self.rewards.append(r)

    def train(self):
        self.actor_critic.train()
        reward = torch.FloatTensor(self._discount_and_norm_rewards()).detach()
        self.loglikelihood = torch.stack(self.loglikelihood)
        self.state_values = torch.stack(self.state_values)
        
        value_loss = self.loss_func(reward,self.state_values)
        actor_loss = torch.mean(-self.loglikelihood * (reward-self.state_values))
        self.optimizer.zero_grad()
        loss = value_loss + actor_loss
        loss.backward()
        self.optimizer.step()

        self.state_values, self.rewards, self.loglikelihood = [], [], []
        return loss.data.item()

    def _discount_and_norm_rewards(self):
        discount = np.zeros_like(self.rewards)
        tmp = 0
        for i in reversed(range(len(self.rewards))):
            tmp = tmp * GAMMA + self.rewards[i]
            discount[i] = tmp
        discount = discount - np.mean(discount)
        discount = discount / (np.std(discount)+1e-8)
        return discount

# hyperparameters
env_name = 'CartPole-v0'                     # you can try 'LunarLander-v2'
LR = 0.003                                   # learning rate
GAMMA = 0.9                                  # reward discount
env = gym.make(env_name).unwrapped           # unwrapped gym simulation CartPole
env.seed(1)                                  # reproducible, Policy gradient has high variance
N_ACTIONS = env.action_space.n               # discrete action space with 2 actions
N_STATES = env.observation_space.shape[0]    # continuous state
print(env.action_space.n,env.observation_space)

model_path = './checkpoint/a2c.pth'
a2c = A2C(N_ACTIONS, N_STATES, model_path)

losses, rewards = [], []
for i_episode in range(300):
    s = env.reset()
    while True:
        env.render()
        a = a2c.choose_action(s)
        s_, r, done, _ = env.step(a)
        a2c.store_transition(r)
        s = s_
        if done:
            ep_rs_sum = sum(a2c.rewards)
            if 'episode_reward_sum' not in globals():
                episode_reward_sum = ep_rs_sum
            else:
                episode_reward_sum = episode_reward_sum * 0.99 + ep_rs_sum * 0.01
            print("episode: %d, reward: %.2f"%(i_episode, episode_reward_sum))
            rewards.append(episode_reward_sum)
            losses.append(a2c.train())
            break


torch.save(a2c.actor_critic.state_dict(),model_path)
plt.subplot(1,2,1)
plt.title('training loss')
plt.plot(losses)
plt.grid()
plt.subplot(1,2,2)
plt.title('long-term reward')
plt.plot(rewards)
plt.grid()
plt.show()
