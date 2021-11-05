import gym

# refer to https://www.jianshu.com/p/e7235f8af25e

register_envs = gym.envs.registry.all()
for e in register_envs: print(e)

# discrete action space
env = gym.make('CartPole-v0').unwrapped        # 使用gym库中的环境：CartPole，且打开封装
N_ACTIONS = env.action_space.n                 # 杆子动作个数 (2个)
N_STATES = env.observation_space.shape[0]      # 杆子状态维数 (4个)
print('CartPole-v0:\n',env.action_space,env.observation_space)
# state = [x, theta, dx, dtheta]
# action = 0 or 1 (left or right)

s = env.reset()
for i in range(100):
    env.render()
    a = env.action_space.sample()
    s, r, done, _ = env.step(a)
    if done: break
env.close()

# continuous action space
env = gym.make('Pendulum-v1').unwrapped
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0]
print('Pendulum-v1:\n',env.action_space,env.observation_space)
# state = [cos(theta),sin(theta),dtheta], -8.0<=theta dot<=8.0
# action = [joint effort(torque)], -2.0<=torque<=2.0
# reward = -(theta**2+0.1*dtheta**2+0.001*torque**2)

s = env.reset()
for i in range(100):
    env.render()
    a = env.action_space.sample()
    print(a)
    s, r, done, _ = env.step(a)
    if done: break
env.close()


# the simulations below need 'pip install Box2D'
# discrete action space
env = gym.make('LunarLander-v2').unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
print('LunarLander-v2:\n',env.action_space, env.observation_space)
# state = a vector of 8 elements, float32
# action = 0,1,2,3 (discrete)

s = env.reset()
for i in range(100):
    env.render()
    a = env.action_space.sample()
    s, r, done, _ = env.step(a)
    if done: break
env.close()


# continuous action space
env = gym.make('LunarLanderContinuous-v2').unwrapped
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0]
print('LunarLanderContinuous-v2:\n',env.action_space, env.observation_space)
# state = a vector of 8 elements, float32
# action = [-1. -1.], [1. 1.], (2,), float32

s = env.reset()
for i in range(100):
    env.render()
    a = env.action_space.sample()
    print(a)
    s, r, done, _ = env.step(a)
    if done: break
env.close()

# the state is an image, so CNN is needed for DRL
env = gym.make('CarRacing-v0').unwrapped
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0]
print('CarRacing-v0:\n',env.action_space, env.observation_space.shape)
# state = numpy.ndarray (96, 96, 3), uint8
# action = [-1.  0.  0.], [1. 1. 1.], (3,), float32

s = env.reset()
for i in range(100):
    env.render()
    a = env.action_space.sample()
    s, r, done, _ = env.step(a)
    if done: break
env.close()

# many interesting robot simulations need installing MuJoCo
# you can explore them and try to accomplish them with DRL