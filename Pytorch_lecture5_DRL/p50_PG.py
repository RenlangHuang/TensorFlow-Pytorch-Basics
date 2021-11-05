import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

# a more popular implementation than p49
# using torch.distributions to sample an action
# according to the probability distribution
# in the aciton space proposed by the policy network
# and calculate the logarithmic likelihood
# Policy Gradient can be applied to enironments 
# with continuous action space(such as Pendulum-v1) 
# where torch.distributions.MultivariateNormal() 
# is always used to sample actions
# However, PG, AC and A2C are hard to converge!

# for CartPole-v0
class PolicyNet(torch.nn.Module):
    def __init__(self, n_actions, n_states):
        super(PolicyNet, self).__init__()
        self.fcl1 = torch.nn.Linear(n_states, 10)
        self.fcl2 = torch.nn.Linear(10, n_actions)
    def forward(self, x):
        x = torch.tanh(self.fcl1(x))
        return torch.softmax(self.fcl2(x),dim=-1)

# for LunarLander-v2, but not so stable
class PolicyNet2(torch.nn.Module):
    def __init__(self, n_actions, n_states):
        super(PolicyNet, self).__init__()
        self.fcl1 = torch.nn.Linear(n_states, 32)
        self.fcl2 = torch.nn.Linear(32, 32)
        self.fcl3 = torch.nn.Linear(32, n_actions)
    def forward(self, x):
        x = torch.relu(self.fcl1(x))
        x = torch.tanh(self.fcl2(x))
        return torch.softmax(self.fcl3(x),dim=-1)

class PolicyGradient(object):
    def __init__(self, n_actions, n_states, model_path, learning_rate=0.01, discount=0.95, load_pretrained=True):
        self.n_actions = n_actions
        self.n_features = n_states
        self.lr = learning_rate
        self.gamma = discount
        self.rws, self.loglikelihood = [],[]

        self.net = PolicyNet(n_actions, n_states)
        if load_pretrained and os.path.exists(model_path):
            print('------------load the model----------------')
            self.net.load_state_dict(torch.load(model_path))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def choose_action(self, s):
        self.net.eval()
        actions = self.net(torch.Tensor(s[np.newaxis, :]))
        actions = torch.distributions.Categorical(actions)
        action = actions.sample()
        loglikelihood = actions.log_prob(action)
        self.loglikelihood.append(loglikelihood)
        return action.data.item()

    def store_transition(self, r):
        self.rws.append(r)

    def train(self):
        self.net.train()
        reward = torch.FloatTensor(self._discount_and_norm_rewards())
        reward = reward.reshape([len(self.loglikelihood),1])
        self.loglikelihood = torch.stack(self.loglikelihood)
        loss = torch.mean(-self.loglikelihood * reward)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.rws, self.loglikelihood = [],[]
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
env_name = 'CartPole-v0'                      # you can try 'LunarLander-v2' with PolicyNet2
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
        pg.store_transition(r)
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
s = env.close()