import ipdb
import torch 
from utils import *
import numpy as np

################################################################
def getTrajectories(net,env):
    batch_size=2
    traj_list=[]
    actions_list=[]
    reward_list=[]
    for _ in range(batch_size):
        traj,actions,total_reward = sampleTrajectory(net,env)
        ipdb.set_trace()
        traj_list.append(traj) 
        actions_list.append(actions_list)
        reward_list.append(total_reward)
    return np.array(traj_list),np.array(actions_list),np.array(reward_list)

#########################
def sampleTrajectory(net,env):
    traj=[]
    actions=[]

    state= env.reset()
    traj.append(state)
    total_reward = 0
    while True:
        action = getAction(net,state) 
        actions.append(action)

        state, reward, done, info = env.step(action)
        total_reward += reward

        if done == True:
            assert len(traj) == len(actions), "Unequal states and actions!"
            return traj,actions,total_reward
        else:
            traj.append(state)

def getAction(net,state):
    state = numpyFormat(state).float()

    output = net(state).data.numpy()
    return np.argmax(output)
#########################
################################################################
