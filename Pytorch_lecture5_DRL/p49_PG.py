import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt


class PolicyNet(torch.nn.Module):
    def __init__(self, n_actions, n_states, n_hiddens=10):
        super(PolicyNet, self).__init__()
        self.fcl1 = torch.nn.Linear(n_states, n_hiddens)
        self.fcl2 = torch.nn.Linear(n_hiddens, n_actions)
    def forward(self, x):
        x = torch.tanh(self.fcl1(x))
        return torch.softmax(self.fcl2(x),dim=-1)
 

class PolicyGradient(object):
    def __init__(self, n_actions, n_states, model_path, learning_rate=0.01, discount=0.95, load_pretrained=True):
        self.n_actions = n_actions
        self.n_features = n_states
        self.lr = learning_rate
        self.gamma = discount
        self.obs, self.acs, self.rws = [], [], []

        self.net = PolicyNet(n_actions, n_states)
        if load_pretrained and os.path.exists(model_path):
            print('------------load the model----------------')
            self.net.load_state_dict(torch.load(model_path))
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def choose_action(self, s):
        self.net.eval()
        actions = self.net(torch.Tensor(s[np.newaxis, :]))
        action = np.random.choice(range(actions.shape[1]), p=actions.view(-1).detach().numpy())
        return action

    def store_transition(self, s, a, r):
        self.obs.append(s)
        self.acs.append(a)
        self.rws.append(r)

    def train(self):
        self.net.train()
        reward = self._discount_and_norm_rewards()
        output = self.net(torch.Tensor(self.obs))
        one_hot = torch.zeros(len(self.acs), self.n_actions).\
            scatter_(1, torch.LongTensor(self.acs).view(-1,1), 1)
        neg = torch.sum(-torch.log(output) * one_hot, 1)
        loss = neg * torch.Tensor(reward)
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.obs, self.acs, self.rws = [], [], []
        return loss.data.item()

    def _discount_and_norm_rewards(self):
        discount = np.zeros_like(self.rws)
        tmp = 0
        for i in reversed(range(len(self.rws))):
            tmp = tmp * self.gamma + self.rws[i]
            discount[i] = tmp
        discount -= np.mean(discount)
        discount /= np.std(discount)
        return discount

# hyperparameters
env_name = 'CartPole-v0'
LR = 0.01                                     # learning rate,0.02
GAMMA = 0.9                                   # reward discount
env = gym.make(env_name).unwrapped            # unwrapped gym simulation CartPole
env.seed(1)                                   # reproducible, Policy gradient has high variance
N_ACTIONS = env.action_space.n                # discrete action space with 2 actions
N_STATES = env.observation_space.shape[0]     # continuous state
print(env.action_space.n,env.observation_space)

model_path = './checkpoint/pg_'+env_name+'.pth'
pg = PolicyGradient(N_ACTIONS, N_STATES, model_path, LR, GAMMA)
losses, rewards = [], []
for i_episode in range(300):
    s = env.reset()
    while True:
        env.render()
        a = pg.choose_action(s)
        s_, r, done, _ = env.step(a)
        pg.store_transition(s, a, r)
        s = s_
        if done:
            ep_rs_sum = sum(pg.rws)
            if 'episode_reward_sum' not in globals():
                episode_reward_sum = ep_rs_sum
            else:
                episode_reward_sum = episode_reward_sum * 0.99 + ep_rs_sum * 0.01
            print("episode: %d, reward: %.2f"%(i_episode, episode_reward_sum))
            rewards.append(episode_reward_sum)
            losses.append(pg.train())
            break


torch.save(pg.net.state_dict(),model_path)
plt.subplot(1,2,1)
plt.title('training loss')
plt.plot(losses)
plt.grid()
plt.subplot(1,2,2)
plt.title('long-term reward')
plt.plot(rewards)
plt.grid()
plt.show()


# application
s = env.reset()
while True:
    env.render()
    with torch.no_grad():
        a = pg.choose_action(s)
    s, _, done, _ = env.step(a)
    if done: break