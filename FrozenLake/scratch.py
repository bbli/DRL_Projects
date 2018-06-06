import gym

env = gym.make('FrozenLake-v0')
observation = env.reset()
env.render()
print(env.observation_space) # discrete finite observations
