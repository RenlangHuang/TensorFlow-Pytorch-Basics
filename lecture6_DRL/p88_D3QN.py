import os
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class Q_Network(Model):
    def __init__(self, observation_n, action_n):
        super(Q_Network, self).__init__()
        self.observation_n = observation_n
        self.action_n = action_n
        self.fcl = Dense(64, activation="relu")
        self.V_value = Dense(1)
        self.Advantage = Dense(self.action_n)
    
    def call(self, x):
        x = self.fcl(x)
        v = self.V_value(x)
        advantage = self.Advantage(x)
        actions_value = v + advantage - tf.reduce_mean(advantage)
        return actions_value


class DuelingDoubleDQN(object):
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9,
            replace_target_iter=300, memory_size=500, batch_size=32):
        """
        :param n_actions: 动作种类个数
        :param n_features: 观察向量的维度
        :param learning_rate: 学习率
        :param reward_decay: 奖励衰减系数
        :param replace_target_iter: 多少步替换依次权重
        :param memory_size: 经验池的大小
        :param batch_size: 神经网络训练的批样本数
        """
        self.memory_counter = 0
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))

        self.q_target = Q_Network(observation_n=n_features, action_n=self.n_actions)
        self.q_eval = Q_Network(observation_n=n_features, action_n=self.n_actions)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.loss = tf.losses.MeanSquaredError()

    def choose_action(self, state, eps=0.):
        state = tf.Variable(state, dtype=tf.float32)
        if len(tf.shape(state)) == 1:
            state = tf.expand_dims(state, axis=0)
        if np.random.uniform() > eps:
            action_value = self.q_eval.predict(state)
            return np.argmax(action_value)
        else:
            return np.random.choice(np.arange(self.n_actions))

    def train(self):
        # update the target network asynchronously
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.soft_update(self.q_target,self.q_eval,1.0)

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        experience = self.memory[sample_index, :]
        states = np.array([e[:self.n_features] for e in experience]).astype("float32")
        actions = np.array([e[self.n_features] for e in experience]).astype("int32")
        rewards = np.array([e[self.n_features + 1] for e in experience]).astype("float32")
        next_states = np.array([e[-self.n_features:] for e in experience]).astype("float32")
        dones = np.array([e[self.n_features+2] for e in experience])

        next_actions = self.q_eval.predict(next_states)
        next_actions = tf.argmax(next_actions, axis=-1)
        next_actions = list(enumerate(next_actions))

        q_target_values = self.q_target.predict(next_states)
        q_target_values = tf.gather_nd(q_target_values,next_actions)
        q_target_values = rewards + self.gamma * (1 - dones) * tf.squeeze(q_target_values)
        with tf.GradientTape() as tape:
            q_values = self.q_eval(states, training=True)
            enum_actions = list(enumerate(actions))
            q_values = tf.gather_nd(params=q_values, indices=enum_actions)
            loss = self.loss(q_values, q_target_values)

        grads = tape.gradient(loss, self.q_eval.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_eval.trainable_variables))
        return loss

    def store_transition(self, s, a, r, done, s_):
        transition = np.hstack((s, [a, r, done], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def soft_update(self, tar, src, tau=0.01):
        for target_param, local_param in zip(tar.weights, src.weights):
            tf.compat.v1.assign(target_param, tau * local_param + (1. - tau) * target_param)


if __name__ == "__main__":
    # hyper parameters
    BATCH_SIZE = 32
    LR = 0.01                                      # learning rate
    EPSILON = 0.1                                  # greedy policy
    GAMMA = 0.9                                    # reward discount
    TARGET_REPLACE_ITER = 50                       # asynchronous update frequency for the target network
    MEMORY_CAPACITY = 1000                         # experience pool capacity
    EPOCHS = 70                                    # training epochs
    env = gym.make('CartPole-v0').unwrapped        # unwrapped gym simulation CartPole
    N_ACTIONS = env.action_space.n                 # discrete action space with 2 actions
    N_STATES = env.observation_space.shape[0]      # continuous state
    
    # initialize the models and agent
    RL = DuelingDoubleDQN(N_ACTIONS, N_STATES, LR, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, BATCH_SIZE)
    checkpoint_save_path = "./checkpoint/D3QN.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        RL.q_eval.load_weights(checkpoint_save_path)
        RL.q_target.load_weights(checkpoint_save_path)

    # training process
    loss, rewards = [], []
    for episode in range(EPOCHS):
        episode_reward_sum = 0
        # initial observation
        s = env.reset()
        while True:
            env.render()
            # RL choose action based on observation
            a = RL.choose_action(s,EPSILON)
            # RL take action and get next observation and reward
            s_, r, done, _ = env.step(a)
            RL.store_transition(s, a, r, done, s_)
            episode_reward_sum += r

            # training with experience replay
            if RL.memory_counter > BATCH_SIZE:
                loss.append(RL.train())

            s = s_ # update state
            # break while loop when end of this episode
            if done:
                print('episode %d, reward_sum: %.2f' % (episode, episode_reward_sum))
                rewards.append(episode_reward_sum)
                break
    
    # end of training
    print('training over')
    env.close()

    # save the model and visualize the training
    RL.q_eval.save_weights(checkpoint_save_path)
    plt.subplot(1,2,1)
    plt.title('training loss')
    plt.plot(loss)
    plt.grid()
    plt.subplot(1,2,2)
    plt.title('model evaluation')
    plt.plot(rewards,label='long-term reward')
    plt.legend()
    plt.grid()
    plt.show()

    # application
    s = env.reset()
    episode_reward_sum = 0
    while True:
        env.render()
        a = RL.choose_action(s)
        s, r, done, _ = env.step(a)
        episode_reward_sum += r
        if done: break
    print('reward=',episode_reward_sum)