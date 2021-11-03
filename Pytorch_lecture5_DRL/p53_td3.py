import os
import gym
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:',device)

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


# hyperparameters
start_timesteps = 5     # Time steps initial random policy is used
eval_freq = 200         # How often (time steps) we evaluate
MAX_EPISODES = 100      # Max time steps to run environment
MAX_EP_STEPS = 200
expl_noise = 0.1        # Std of Gaussian exploration noise
BATCH_SIZE = 256        # Batch size for both actor and critic
discount = 0.99         # Discount factor
tau = 0.005             # Target network update rate
policy_noise = 0.2      # Noise added to target policy during critic update
noise_clip = 0.5        # Range to clip target policy noise
policy_freq = 2         # Frequency of delayed policy updates
MEMORY_CAPACITY = 10000 # capacity of the replay buffer
LR_A = 0.001            # learning rate for actor
LR_C = 0.002            # learning rate for critic


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(self, s_dim, a_dim, max_action, model_path, discount=0.99, tau=0.005,
        policy_noise=0.2,  noise_clip=0.5, policy_freq=2, load_pretrained=True):
        
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, max_action
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim*2 + a_dim + 1), dtype=np.float32)
        self.actor = Actor(s_dim, a_dim, max_action).to(device)
        self.critic = Critic(s_dim, a_dim).to(device)

        if load_pretrained and os.path.exists(model_path[0]):
            print('------------load the model----------------')
            self.actor.load_state_dict(torch.load(model_path[0]))
            self.critic.load_state_dict(torch.load(model_path[1]))

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_A) #3e-4
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_C)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.memory_counter = 0
        self.total_it = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % MEMORY_CAPACITY # replace the old memory
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def soft_update(self,target,source,epsilon=0.1):
        for target_param, source_param in zip(target.parameters(),source.parameters()):
            target_param.data.copy_((1-epsilon)*target_param.data + epsilon*source_param.data)

    def train(self):
        self.total_it += 1
        if self.total_it==1: print('Begin training!')
        # sample a minbatch from the experience pool (replay buffer)
        if self.memory_counter<MEMORY_CAPACITY:
            sample_index = np.random.choice(self.memory_counter, BATCH_SIZE)
        else: sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        bt = self.memory[sample_index, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim]).to(device)
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim]).to(device)
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim]).to(device)
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:]).to(device)

        with torch.no_grad():
            # choose action according to policy and add clipped noise
            noise = (torch.randn_like(ba)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(bs_) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(bs_, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = br + self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(bs, ba)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        actor_loss = -self.critic.Q1(bs, self.actor(bs)).mean()
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            # actor_loss = -self.critic.Q1(bs, self.actor(bs)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            self.soft_update(self.critic_target, self.critic, self.tau)
            self.soft_update(self.actor_target, self.actor, self.tau)
        
        return [actor_loss.data.item(),critic_loss.data.item()]


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, eval_episodes=10):
	eval_env = gym.make(env_name)
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.choose_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
	avg_reward /= eval_episodes
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
    env = gym.make('Pendulum-v1').unwrapped
    env.seed(1) # facilitate the repetition
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    model_path = ('./checkpoint/td3_actor.pth','./checkpoint/td3_critic.pth')

    kwargs = {
        "model_path": model_path,
        "s_dim": state_dim,
        "a_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
    }

    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = policy_noise * max_action
    kwargs["noise_clip"] = noise_clip * max_action
    kwargs["policy_freq"] = policy_freq
    td3 = TD3(**kwargs)

    # Evaluate untrained td3
    evaluations = [eval_policy(td3,'Pendulum-v1')]

    var = 3.0  # control exploration
    losses, rewards = [], []
    for i in range(MAX_EPISODES):
        s = env.reset()
        episode_reward_sum = 0
        for j in range(MAX_EP_STEPS):
            env.render()
            # add exploration noise
            a = td3.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)
            s_, r, _, _ = env.step(a)

            # Store data in replay buffer
            td3.store_transition(s, a, r/10, s_)

            s = s_

            if td3.memory_counter > MEMORY_CAPACITY:
                if var>0.03: var *= 0.9995 # decay the action randomness
                losses.append(td3.train())
            episode_reward_sum += r

        print('episode %d, reward_sum: %.2f, explore: %.2f' % (i, episode_reward_sum, var))
        rewards.append(episode_reward_sum/MAX_EP_STEPS)

        # Evaluate episode
        if (i + 1) % eval_freq == 0:
            evaluations.append(eval_policy(td3, 'Pendulum-v1'))

    torch.save(td3.actor.state_dict(),model_path[0])
    torch.save(td3.critic.state_dict(),model_path[1])

    # application
    s = env.reset()
    while True:
        env.render()
        with torch.no_grad():
            a = td3.choose_action(s)
        s, _, done, _ = env.step(a)
        if done: break
