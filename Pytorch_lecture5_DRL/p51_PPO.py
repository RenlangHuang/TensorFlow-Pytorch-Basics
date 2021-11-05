import os
import gym
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Implementation of Proximal Policy Optimization (PPO)
# Paper: Proximal Policy Optimization Algorithms (2017), https://arxiv.org/abs/1707.06347
# refer to https://zhuanlan.zhihu.com/p/359306335


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:',device)

class ActorNet(torch.nn.Module):
    def __init__(self, n_states, n_actions, bound):
        super(ActorNet, self).__init__()
        self.bound = bound
        self.layer = torch.nn.Linear(n_states, 128)
        self.mean_out = torch.nn.Linear(128, n_actions)
        self.var_out = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer(x))
        mean = self.bound * torch.tanh(self.mean_out(x))
        var = F.softplus(self.var_out(x))
        return mean, var


class CriticNet(torch.nn.Module):
    def __init__(self, n_states):
        super(CriticNet, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(n_states, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.layer(x)


class PPO(object):
    def __init__(self, n_states, n_actions, bound, model_path, lr=1e-4, gamma=0.9, epsilon=0.2,
        actor_update_steps = 10, critic_update_steps = 10, load_pretrained=True):
        self.n_states = n_states
        self.n_actions = n_actions
        self.bound = bound
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.a_update_steps = actor_update_steps
        self.c_update_steps = critic_update_steps

        self.actor_model = ActorNet(n_states, n_actions, bound).to(device)
        self.actor_old_model = ActorNet(n_states, n_actions, bound).to(device)
        self.actor_optim = torch.optim.Adam(self.actor_model.parameters(), lr=self.lr)

        self.critic_model = CriticNet(n_states).to(device)
        self.critic_optim = torch.optim.Adam(self.critic_model.parameters(), lr=self.lr)
        self.loss_func = torch.nn.MSELoss()

        if load_pretrained and os.path.exists(model_path[0]):
            print('------------load the model----------------')
            self.actor_model.load_state_dict(torch.load(model_path[0]))
            self.critic_model.load_state_dict(torch.load(model_path[1]))

    def choose_action(self, s):
        s = torch.FloatTensor(s).to(device)
        mean, var = self.actor_model(s)
        actions = torch.distributions.Normal(mean, var)
        action = actions.sample().cpu()
        return np.clip(action, -self.bound, self.bound)

    def discount_reward(self, rewards, s_):
        s_ = torch.FloatTensor(s_).to(device)
        target = self.critic_model(s_).detach()
        target_list = []
        for r in rewards[::-1]:
            target = r + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        return torch.cat(target_list)

    def actor_learn(self, states, actions, advantage):
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).reshape(-1, 1).to(device)

        mu, sigma = self.actor_model(states)
        pi = torch.distributions.Normal(mu, sigma)

        old_mu, old_sigma = self.actor_old_model(states)
        old_pi = torch.distributions.Normal(old_mu, old_sigma)

        # importance sampling factor (pi's likelihood/pi_old's likelihood)
        ratio = torch.exp(pi.log_prob(actions) - old_pi.log_prob(actions))
        surr = ratio * advantage.reshape(-1, 1)
        # PPO with Clipped Objective (easier than PPO with Adaptive KL Penalty)
        loss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantage.reshape(-1, 1)))

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, states, targets):
        states = torch.FloatTensor(states).to(device)
        v = self.critic_model(states).reshape(1, -1).squeeze(0)
        loss = self.loss_func(v, targets)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def cal_advantage(self, states, targets):
        states = torch.FloatTensor(states).to(device)
        v = self.critic_model(states)
        advantage = targets - v.reshape(1, -1).squeeze(0)
        return advantage.detach()

    def train(self, states, actions, targets):
        self.actor_old_model.load_state_dict(self.actor_model.state_dict())
        advantage = self.cal_advantage(states, targets)

        for i in range(self.a_update_steps):
            self.actor_learn(states, actions, advantage)

        for i in range(self.c_update_steps):
            self.critic_learn(states, targets)


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    env.seed(10)
    torch.manual_seed(10)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    bound = env.action_space.high[0]
    model_path = ('./checkpoint/ppo_actor.pth','./checkpoint/ppo_critic.pth')
    ppo = PPO(n_states, n_actions, bound, model_path)

    rewards = []
    for i_episode in range(700):
        ep_rs_sum = 0
        s = env.reset()
        states, actions, rewards = [], [], []
        for t in range(200):
            #env.render()
            a = ppo.choose_action(s)
            s_, r, done, _ = env.step(a)
            ep_rs_sum += r
            states.append(s)
            actions.append(a)
            rewards.append((r + 8) / 8)
            s = s_

            if (t + 1) % 32 == 0 or t == 199: # batch_size = 32
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)

                targets = ppo.discount_reward(rewards, s_)
                ppo.train(states, actions, targets)
                states, actions, rewards = [], [], []

        if 'episode_reward_sum' not in globals():
            episode_reward_sum = ep_rs_sum
        else:
            episode_reward_sum = episode_reward_sum * 0.99 + ep_rs_sum * 0.01
        print("episode: %d, reward: %.2f"%(i_episode, episode_reward_sum))
        rewards.append(episode_reward_sum)
    
    plt.plot(rewards)
    plt.show()
    
    torch.save(ppo.actor_model.state_dict(),model_path[0])
    torch.save(ppo.critic_model.state_dict(),model_path[1])

    # application
    s = env.reset()
    while True:
        env.render()
        with torch.no_grad():
            a = ppo.choose_action(s)
        s, _, done, _ = env.step(a)
        if done: break
    s = env.close()