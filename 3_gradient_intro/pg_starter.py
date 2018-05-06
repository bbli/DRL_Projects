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
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter
import ipdb

from utils import *
from functions import *


def trainModel(probability,neurons):
    ################ **Defining Model and Environment** ##################

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(6,neurons)
            self.dropout = nn.DropOut(p=probability)
            self.fc2 = nn.Linear(neurons,3)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return F.softmax(self.fc2(x))

    net = Net()
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    # w = SummaryWriter()
    count = 0
    env = gym.make('Acrobot-v1')
    # print(env.action_space)
    # print(env.observation_space)
    # showModel(net)
    # randomWalk()

    ################################################################
    num_episodes = 1200
    baseline = -500
    num_trajectory = 16
    for episode in range(num_episodes):
        # print(episode)
        before_weights_list = weightMag(net)
        ################# **Evaluating the Loss across Trajectories** ###################
        for i in range(num_trajectory):
            count +=1
            trajectory, actions, reward = sampleTrajectory(net,env)
            # w.add_scalar('Reward',reward,count)
            probs = net(numpyFormat(trajectory).float())
            ipdb.set_trace()

            traj_loss = LogLoss(probs,actions,reward,baseline)
            baseline = 0.99*baseline + 0.01*reward
            if i == 0:
                total_loss = traj_loss
            else:
                total_loss += traj_loss
        ## Averaging to create loss estimator
        total_loss = torch.mul(total_loss,1/num_trajectory)
        # w.add_scalar('Loss', total_loss.data[0],episode)
        ################################################################
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        after_weights_list =weightMag(net)
        relDiff_list = relDiff(before_weights_list,after_weights_list)
        relDiff_dict = listToDict(relDiff_list)
        # w.add_scalars('LayerChanges',relDiff_dict,count)
    # w.close()
    return net
    ################################################################

decay_parameters = [0,1e-4,1e-3]
num_traj_parameters = [10,15,20]
neuron_parameters = [10,20,30,40]
min_runs = 500
run_count =0
for decay in decay_parameters:
    for num in num_traj_parameters:
        for neuron in neuron_parameters:
            run_count +=1
            print("Run {}",run_count)
            model = trainModel(decay, num, neuron)
            # average_runs = evaluateModel(model)
            average_runs, std = averageModelRuns(model)
            print("Decay: {} Number of Trajectories: {} Hidden Units: {}".format(decay, num, neuron))
            print("Mean runs: {}, Standard Deviation: {}".format(average_runs,std))
            if average_runs<min_runs:
                best_model = model
                min_runs = average_runs

torch.save(best_model.state_dict(),'best_model.pt')
