#==========================================================================  
# Policy Gradient Starter Code 
# 
#==========================================================================

"""
https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py

Acrobot is a 2-link pendulum with only the second joint actuated
Intitially, both links point downwards. The goal is to swing the
end-effector at a height at least the length of one link above the base.
Both links can swing freely and can pass by each other, i.e., they don't
collide when they have the same angle.

**STATE:**
The state consists of the sin() and cos() of the two rotational joint
angles and the joint angular velocities :
[cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
For the first link, an angle of 0 corresponds to the link pointing downwards.
The angle of the second link is relative to the angle of the first link.
An angle of 0 corresponds to having the same angle between the two links.
A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.

**ACTIONS:**
The action is either applying +1, 0 or -1 torque on the joint between
the two pendulum links.
"""

"""
In order to render the env, you need to run the following command:
    pip install cython 'gym[classic_control]'
"""

import gym
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import ipdb

from utils import *
from functions import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6,16)
        self.fc2 = nn.Linear(16,3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x))

net = Net()
count = 0
# w = SummaryWriter()
env = gym.make('Acrobot-v1')
# print(env.action_space)
# print(env.observation_space)
evaluateModel(net)
randomWalk()

num_episodes = 2000
num_trajectory = 10
for episode in range(num_episodes):
    count +=1

    ################### **Evaluating the Loss across Trajectories** #########################
    trajectory, actions, reward = sampleTrajectory(net,env)
    w.add_scalar('Reward',reward,count)
    probs = net(numpyFormat(trajectory).float())
    total_loss = LogLoss(probs,actions,reward)
    print(total_loss)

    for _ in range(num_trajectory-1):
        count +=1
        trajectory, actions, reward = sampleTrajectory(net,env)
        w.add_scalar('Reward',reward,count)
        probs = net(numpyFormat(trajectory).float())

        traj_loss = LogLoss(probs,actions,reward)
        total_loss += traj_loss
        print(total_loss)
    ## Averaging to create loss estimator
    total_loss = torch.mul(total_loss,1/num_trajectory)
    ipdb.set_trace()
    w.add_scalar('Loss', total_loss.data[0],episode)
    ################################################################
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

evaluateModel(net)
