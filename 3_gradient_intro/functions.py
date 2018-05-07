import ipdb
import torch 
from utils import *
import numpy as np
import getch
import gym
import time

################################################################
def getTrajectories(net,env):
    batch_size=2
    trajectory_list=[]
    actions_list=[]
    reward_list=[]
    for _ in range(batch_size):
        trajectory,actions,total_reward = sampleTrajectory(net,env)
        # ipdb.set_trace()
        trajectory_list.append(trajectory) 
        actions_list.append(actions_list)
        reward_list.append(total_reward)
    return np.array(trajectory_list),np.array(actions_list),np.array(reward_list)

#########################
def sampleTrajectory(net,env,last_reward):
    trajectory=[]
    actions=[]

    state= env.reset()
    trajectory.append(state)
    total_reward = 0
    while True:
        if abs(last_reward)<80:
            action = getAction(net,state,True) 
        else:
            action = getAction(net,state)
        actions.append(action)

        state, reward, done, info = env.step(action)
        total_reward += reward

        if done == True:
            assert len(trajectory) == len(actions), "Unequal states and actions!"
            return np.array(trajectory),np.array(actions),total_reward
        else:
            trajectory.append(state)

def getAction(net,state,force=False):
    state = numpyFormat(state).float()

    output = net(state).data.numpy()
    action = np.random.choice([0,1,2],p=output)
    if force:
        action = np.argmax(output)
    return action

#########################
################################################################

def LogLoss(probs, actions, reward, baseline):
    for i,(prob,action) in enumerate(zip(probs,actions)):
        if i ==0:
            total_loss = torch.log(prob[action])
        else:
            total_loss += torch.log(prob[action])
    # negative in front since optimizer does gradient descent
    total_loss = torch.mul(total_loss,-1)
    return torch.mul(total_loss,reward-baseline)

def showModel(net):
    print("This is sampling from the untrained network")
    env = gym.make('Acrobot-v1')
    state = env.reset()
    count =0
    while True:
        count += 1
        env.render()
        action=getAction(net,state)
        state,reward,done,info = env.step(action)
        time.sleep(0.04)
        if done == True:
            print("Click any key to close the environment")
            getch.getch()
            env.close()
            return count

def randomWalk():
    env = gym.make('Acrobot-v1')
    print("This is just random action sampling")
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        time.sleep(0.04)
        if done == True:
            print("Click any key to close the environment")
            getch.getch()
            env.close()
            return 0
################################################################
def averageModelRuns(model):
    env = gym.make('Acrobot-v1')
    num_trials = 100
    counts_list=[]
    for _ in range(num_trials):
        state = env.reset()
        count = evaluateModel(model)
        counts_list.append(count)
    counts_list = np.array(counts_list)
    return counts_list.mean(), counts_list.std(ddof=1)

def evaluateModel(net):
    env = gym.make('Acrobot-v1')
    state = env.reset()
    net.eval()
    count =0
    while True:
        count += 1
        action=getAction(net,state)
        state,reward,done,info = env.step(action)
        if done == True:
            return count
