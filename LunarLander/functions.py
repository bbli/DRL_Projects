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
        if char == 'w':
            return 0
        elif char == 'a':
            return 1
        elif char == 's':
            return 2
        elif char == 'd':
            return 3
        elif char == 'b':
            return 'b'



################ **Loss all at Once** ##################
def LogLoss(probs, actions, reward, baseline):
    for i,(prob,action) in enumerate(zip(probs,actions)):
        if i ==0:
            total_loss = torch.log(prob[action])
        else:
            total_loss += torch.log(prob[action])
    # negative in front since optimizer does gradient descent
    total_loss = torch.mul(total_loss,-1)
    return torch.mul(total_loss,reward-baseline)

def baselineTune(probs,actions,reward,baseline,episode):
    if (episode>400) & (abs(reward)>abs(baseline)) & (baseline<100):
        return LogLoss(probs,actions,baseline,baseline)
    else:
        return LogLoss(probs,actions,reward,baseline)

def sampleTrajectory(net,env):
    trajectory=[]
    actions=[]

    state= env.reset()
    trajectory.append(state)
    total_reward = 0
    while True:
        action = getAction(net,state) 
        actions.append(action)

        state, reward, done, info = env.step(action)
        total_reward += reward

        if done == True:
            assert len(trajectory) == len(actions), "Unequal states and actions!"
            return np.array(trajectory),np.array(actions),total_reward
        else:
            trajectory.append(state)

def getTrajectoryLoss(net,env,count,baseline,episode,num_trajectory,w=None):
    local_count =count
    for i in range(num_trajectory):
        local_count +=1
        trajectory, actions, reward = sampleTrajectory(net,env)
        probs = net(numpyFormat(trajectory).float())
        # traj_loss = baselineTune(probs,actions,reward,baseline,episode)
        traj_loss = LogLoss(probs,actions,reward,baseline)
        
        baseline = 0.99*baseline + 0.01*reward
        if i == 0:
            total_loss = traj_loss
        else:
            total_loss += traj_loss

        ################ **Logging** ##################
        if w:
            w.add_scalar('Reward',reward,local_count)
            w.add_scalar('Baseline',baseline,local_count)
        ##############################################################
    # if total_loss.data[0] == 0:
        # ipdb.set_trace()
    ## Averaging to create loss estimator
    total_loss = torch.mul(total_loss,1/num_trajectory)
    ################ **More Logging** ##################
    
    if w:
        w.add_scalar('Loss', total_loss.data[0],episode)
    return total_loss,local_count,baseline
################################################################



################ **Loss on the Go** ##################
################################################################
def getNodesAndReward(net,env):
    # no traj list b/c I am not passing back into net again
    # no actions list b/c I already used it to create a loss graph
    # Yes reward
    '''
    Returns: total reward
    Returns: list of output nodes, already filterd by the action
    '''
    state = env.reset()
    total_reward =0
    output_nodes_list=[]
    while True:
        probs = getOutput(net,state)
        # print(probs)
        action = getOutputAction(probs)
        node = probs[action]
        # if node.data[0] == 1:
            # ipdb.set_trace()
        output_nodes_list.append(node)

        state,reward,done,info = env.step(action)
        total_reward += reward
        if done == True:
            return output_nodes_list,total_reward
################################################################


def getLogLoss(nodes_list,reward,baseline):
    for i,node in enumerate(nodes_list):
        if i == 0:
            traj_loss = torch.log(node)
        else:
            traj_loss += torch.log(node)
    # negative in front since optimizer does gradient descent
    traj_loss = torch.mul(traj_loss,-1)
    return torch.mul(traj_loss,reward-baseline)

def getBaselineTune(nodes_list,reward,baseline,episode):
    if (reward>baseline) & (episode>600) & (baseline>0):
        return getLogLoss(nodes_list,baseline,baseline)
    else:
        return getLogLoss(nodes_list,reward,baseline)
################################################################
    
def getTotalLoss(net,env,count,baseline,episode,num_trajectory,w=None):
    local_count =count
    for i in range(num_trajectory):
        local_count +=1

        nodes_list, reward = getNodesAndReward(net,env)
        # print("Reward-Baseline: ",reward-baseline)
        traj_loss = getLogLoss(nodes_list,reward,baseline)
        # traj_loss = getBaselineTune(nodes_list,reward,baseline,episode)
        
        baseline = 0.95*baseline + 0.05*reward
        if i == 0:
            total_loss = traj_loss
        else:
            total_loss += traj_loss

        ################ **Logging** ##################
        if w:
            w.add_scalar('Reward',reward,local_count)
            w.add_scalar('Baseline',baseline,local_count)
        ##############################################################
    ## Averaging to create loss estimator
    if total_loss.data[0] == 0:
        ipdb.set_trace()
    total_loss = torch.mul(total_loss,1/num_trajectory)
    # print("Total Loss: ",total_loss.data[0])
    ################ **More Logging** ##################
    
    if w:
        w.add_scalar('Loss', total_loss.data[0],episode)
    return total_loss,local_count,baseline

def getSamples(net,env,num_trajectory):
    traj_s_a_list =[]
    traj_nodes_list = []
    for i in range(num_trajectory):
        state = env.reset()
        rewards_list =[]
        states_list = []
        done = False
        while not done:
            states_list.append(state)

            probs = getOutput(net,state)
            action = getOutputAction(probs)
            state,reward,done,info = env.step(action)

            rewards_list.append(reward) 
            traj_nodes_list.append(probs[action])

        rewards_list = np.array(rewards_list)
        rewards_list = np.expand_dims(rewards_list,axis=1)
        states_list = np.array(states_list)

        traj_s_a_list.append((states_list,rewards_list))
    return traj_s_a_list,traj_nodes_list

def createNextStateValue(next_states_list,value_net):
    state = numpyFormat(next_states_list).float()
    output_value = value_net(state).data.numpy()
    length, width = output_value.shape    
    next_state_value = np.zeros((length+1,width))
    next_state_value[:-1] = output_value
    return next_state_value

def createStatesAndTargets(traj_s_a_list,value_net):
    '''
    Returns the states across all trajectories and their corresponding 
    target = r+value_net(next_state)
    '''
    for i,(states_list,rewards_list) in enumerate(traj_s_a_list):
        next_states_list = states_list[1:]
        next_state_value = createNextStateValue(next_states_list,value_net)

        targets = rewards_list + next_state_value
        if i == 0:
            total_states_list = states_list
            total_targets_list = targets
        else:
            total_states_list = np.concatenate((total_states_list,states_list),axis=0)
            total_targets_list = np.concatenate((total_targets_list,targets),axis=0)
    return total_states_list, total_targets_list

def getBootstrappedAdvantageLogLoss(traj_nodes_list,advantage):
    for i,(node,advantage) in enumerate(zip(traj_nodes_list,advantage)):
        advantage = numpyFormat(advantage).float()
        if i == 0:
            total_loss = torch.log(node)*advantage
        else:
            total_loss += torch.log(node)*advantage
    total_loss = torch.mul(total_loss,-1)
    return total_loss


def createAdvantage(traj_s_a_list,critic_net):

    advantage_states , advantage_targets = createStatesAndTargets(traj_s_a_list,critic_net)
    advantage_states = numpyFormat(advantage_states).float()
    advantage = advantage_targets-critic_net.forward(advantage_states).data.numpy()
    return advantage

