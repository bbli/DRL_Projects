#==========================================================================  
# Approximate Q-Learning Starter Code 
# 
# The code was initially written for UCSB Deep Reinforcement Learning Seminar 2018
#
# Authors: Jieliang (Rodger) Luo, Sam Green
#
# April 11th, 2018
#==========================================================================

import gym
import numpy as np
import random
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

num_episodes = 2000
max_steps = 99

epsilon = 0.1 # e-greedy policy 

# build a one-layer network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc1(x)
        return x

def main():

    env = gym.make('FrozenLake-v0')

    print(env.action_space)
    print(env.observation_space) # discrete finite observations

    q_function = Net()

    for episode in range(num_episodes):
        
        # Reset environment and get first new observation
        observation = env.reset()
        rewards = 0
        done = False
        step = 0
        
        #The Q-Network
        while step < max_steps:
                        
            step += 1

            # the following two lines should be commented out during the training to speed the learning process
            env.render()
            time.sleep(1)

            # Choose an action by greedily (with e chance of random action) from the Q-network
            observation_vector = np.identity(16)[observation:observation+1]
            print(observation_vector)

            action_values = q_function(Variable(torch.FloatTensor(observation_vector)))
            print(action_values)
            
            action = np.argmax(action_values.data.numpy())
            print(action)

            if np.random.rand(1) < epsilon:
                action = env.action_space.sample()
            
            #Get new state and reward from environment
            observation_new, reward, done, info = env.step(action)
            
            #Obtain the Q' values by feeding the new state through our network

            
            #Obtain maxQ' and set our target value for chosen action.

            
            #Train our network using target and predicted Q values

            rewards += reward
            observation = observation_new
            
            if done == True:
                
                #Reduce chance of random action as we train the model.
                print("episode {} ends".format(episode))
                print("rewards are {}".format(rewards))

                break

if __name__ == "__main__":
    main()