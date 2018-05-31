import ipdb
import torch 
from utils import *
import numpy as np
import getch
import gym
import time
# import warnings
# warnings.filterwarnings("error")
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import *

def humanInput():
    invalid = True
    while invalid:
        char = getch.getch()
        if char == 'a':
            return 0
        elif char == 's':
            return 1
        elif char == 'd':
            return 2
        elif char == 'b':
            return 'b'


def createQValues(new_states,new_states_actions,critic_net):
    new_states,new_states_actions = numpyFormat(new_states).float(), numpyFormat(new_states_actions).float()
    q_values = critic_net(new_states,new_states_actions).data.numpy()
    return q_values

def createQValueNodes(states,action_nodes,critic_net):
    states = numpyFormat(states).float()
    q_values = critic_net(states,action_nodes)
    return q_values


def createActionNodes(new_states,actor_net):
    new_states = numpyFormat(new_states).float()
    new_optimal_actions = actor_net(new_states)
    return new_optimal_actions

def getMiniBatch(memory_buffer,N):
    '''
    returns multiple numpy arrays
    '''
    states_list = []
    actions_list = []
    rewards_list = []
    new_states_list = []
    memory_buffer_index = np.arange(len(memory_buffer))
    sampled_buffer_index = np.random.choice(memory_buffer_index,N,replace=False)
    for index in sampled_buffer_index:
        state, action, reward, new_state = memory_buffer[index]
        states_list.append(state)
        actions_list.append(action)
        rewards_list.append(reward)
        new_states_list.append(new_state)
    states_list = np.array(states_list)
    new_states_list = np.array(new_states_list)
    actions_list = np.array(actions_list)

    rewards_list = np.array(rewards_list)
    rewards_list = np.expand_dims(rewards_list,axis=1)
    return states_list,actions_list,rewards_list,new_states_list

def updateTargetNetwork(net,target_net):
    tau = 0.95
    net_state_dict = net.state_dict()
    target_net_state_dict = target_net.state_dict()
    # print("Before")
    # print(target_net_state_dict)
    for key in net_state_dict:
        net_tensor = net_state_dict[key]
        target_net_tensor = target_net_state_dict[key]
        smoothed_tensor = tau*target_net_tensor+(1-tau)*net_tensor

        target_net_state_dict[key] = smoothed_tensor
    # print("After")
    # print(target_net_state_dict)
        
