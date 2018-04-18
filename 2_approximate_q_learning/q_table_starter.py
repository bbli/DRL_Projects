#==========================================================================  
# Q-Table Learning Starter Code 
# 
# The code was initially written for UCSB Deep Reinforcement Learning Seminar 2018
#
# Authors: Jieliang (Rodger) Luo, Sam Green
#
# April 17th, 2018
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
import ipdb

num_episodes = 2000
max_steps = 99

epsilon = 0.1 # e-greedy policy 

def main():

    env = gym.make('FrozenLake-v0')

    print(env.action_space)
    print(env.observation_space) # discrete finite observations

    # initialize the table with all zeros
    q_table = np.zeros([env.observation_space.n,env.action_space.n])

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

            # Choose an action by greedily (with e chance of random action) from the Q-table
            action = np.argmax(q_table[observation,:])
            print(action)

            if np.random.rand(1) < epsilon:
                action = env.action_space.sample()
            
            #Get new state and reward from environment
            observation_new, reward, done, info = env.step(action)
            
            # Update Q-Table
            gamma=1
            ipdb.set_trace()
            target = reward + gamma*np.argmax(q_table[observation_new,:])
            q_table[observation,action] = (1-alpha)*q_table[observation,action]+ alpha*target
            
            rewards += reward
            observation = observation_new
            
            if done == True:
                
                #Reduce chance of random action as we train the model.
                print("episode {} ends".format(episode))
                print("rewards are {}".format(rewards))

                break

if __name__ == "__main__":
    main()
