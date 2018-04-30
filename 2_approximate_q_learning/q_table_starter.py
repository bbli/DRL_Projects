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
from utils import *
from functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

num_episodes = 2000
max_steps = 99


env = gym.make('FrozenLake-v0')

print(env.action_space)
print(env.observation_space) # discrete finite observations

# initialize the table with all zeros
q_table = np.zeros([env.observation_space.n,env.action_space.n])
rewards_list =[]
actions_list =[]
observations_list=[]
random_q_table = np.zeros([env.observation_space.n,env.action_space.n])
random_count =0
for episode in range(num_episodes):
    epsilon = 0.4 # e-greedy policy 
    # epsilon = epsilonChooser(episode)
    
    # Reset environment and get first new observation
    observation = env.reset()
    rewards = 0
    done = False
    step = 0
    random_episode =400
    if episode==random_episode:
        print("got Qtable")
        random_q_table = np.copy(q_table)
    
    #The Q-Network
    while step < max_steps:
        step += 1
        # the following two lines should be commented out during the training to speed the learning process
        # env.render()
        # time.sleep(1)

        ################### **Get Action** #########################
        if (np.random.rand(1)<epsilon) and (episode<random_episode):
            action = env.action_space.sample()
            random_count +=1
            getStateandAction(observation,action,observations_list,actions_list) 
        else:
            action = np.argmax(q_table[observation,:])

        ################################################################
        # print("Action: {}".format(action))
        #Get new state and reward from environment
        observation_new, reward, done, info = env.step(action)
        
        # if reward != 0 and episode<250:
            # ipdb.set_trace()
        ################################################################
        # Update Q-Table
        gamma=0.9
        # gamma =1
        alpha = alphaChooser(episode)
        target = reward + gamma*np.max(q_table[observation_new,:])
        q_table[observation,action] = (1-alpha)*q_table[observation,action]+ alpha*target
        # QTableChecker(q_table, episode)
        
        rewards += reward
        observation = observation_new
        actions_list.append(int(action))
        observations_list.append(int(observation))
        
        if done == True:
            
            #Reduce chance of random action as we train the model.
            # print("episode {} ends".format(episode))
            print("rewards are {}".format(rewards))

            break
    rewards_list.append(rewards)

sa_dict = {'state':observations_list,'action':actions_list}
sa_df = pd.DataFrame.from_dict(sa_dict)
# sns.regplot(x='state', y= 'action',data = sa_df)
sns.jointplot(x='state', y= 'action',data = sa_df)
plt.show()
# for episode in range(num_episodes):
    # observation = env.reset()
    # rewards = 0
    # done = False
    # step = 0
    # while step < max_steps:
        # step += 1

        # action = np.argmax(q_table[observation,:])
        # observation_new, reward, done, info = env.step(action)

        # rewards += reward
        # observation = observation_new
        # if done == True:
            
            # #Reduce chance of random action as we train the model.
            # print("episode {} ends".format(episode))
            # print("rewards are {}".format(rewards))

            # break
