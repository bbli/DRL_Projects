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

class Net(nn.Module):
    def __init__(self,neurons):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6,neurons)
        self.fc2 = nn.Linear(neurons,3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x,dim=0)

def generateNetwork(env,neurons):

    # start = time.time()
    while True:
        net = Net(neurons)
        net.train()
        optimizer = optim.Adam(net.parameters(),lr=0.01)
        num_episode=30
        num_trajectory=6
        count=0
        baseline = -500
        for episode in range(num_episode):
            # print(episode)
            total_loss,count,baseline = getTrajectoryLoss(net,env,count,baseline,episode)
            # total_loss,count,baseline = getTotalLoss(net,env,count,baseline,episode)
            updateNetwork(optimizer,total_loss)
            if total_loss.data[0]>1:
                print("Generated Net")
                # end = time.time()
                # print("Elapsed Time: {}".format(end-start))
                return net

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

def getTrajectoryLoss(net,env,count,baseline,episode,w=None):
    num_trajectory=10
    local_count =count
    for i in range(num_trajectory):
        local_count +=1
        trajectory, actions, reward = sampleTrajectory(net,env)
        probs = net(numpyFormat(trajectory).float())
        traj_loss = baselineTune(probs,actions,reward,baseline,episode)
        
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
        action = getOutputAction(probs)
        output_nodes_list.append(probs[action])

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
    if (abs(reward)>abs(baseline)) & (episode>400) & (baseline<100):
        return getLogLoss(nodes_list,baseline,baseline)
    else:
        return getLogLoss(nodes_list,reward,baseline)
################################################################
    
def getTotalLoss(net,env,count,baseline,episode,w=None):
    num_trajectory=16
    local_count =count
    for i in range(num_trajectory):
        local_count +=1

        nodes_list, reward = getNodesAndReward(net,env)
        if abs(reward)<500:
            ipdb.set_trace()
        # traj_loss = getLogLoss(nodes_list,reward,baseline)
        traj_loss = getBaselineTune(nodes_list,reward,baseline,episode)
        
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
    ## Averaging to create loss estimator
    total_loss = torch.mul(total_loss,1/num_trajectory)
    ################ **More Logging** ##################
    
    if w:
        w.add_scalar('Loss', total_loss.data[0],episode)
    return total_loss,local_count,baseline

