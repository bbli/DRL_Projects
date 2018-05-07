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
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
import ipdb

from utils import *
from functions import *


def trainModel(probability,neurons):
    ################ **Defining Model and Environment** ##################

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(6,neurons)
            self.dropout = nn.Dropout(p=probability)
            self.fc2 = nn.Linear(neurons,3)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.softmax(x)

    net = Net()
    net.train()
    w = SummaryWriter()
    count = 0
    env = gym.make('Acrobot-v1')
    # print(env.action_space)
    # print(env.observation_space)
    # showModel(net)
    # randomWalk()

    ################################################################
    num_episodes = 1000
    baseline = -500
    num_trajectory = 16
    optimizer1 = optim.Adam(net.parameters(), lr=0.01)
    optimizer2 = optim.SGD(net.parameters(),  lr=0.001,momentum=0.8)
    scheduler2 = LambdaLR(optimizer2,lr_lambda=cyclic(60))
    optimizer3 = optim.RMSprop(net.parameters(), lr=0.01,alpha=0.95)
    for episode in range(num_episodes):
        # print(episode)
        # before_weights_list = layerMag(net)
        before_weights = netMag(net)
        ################# **Evaluating the Loss across Trajectories** ###################
        for i in range(num_trajectory):
            count +=1
            trajectory, actions, reward = sampleTrajectory(net,env)
            w.add_scalar('Reward',reward,count)
            probs = net(numpyFormat(trajectory).float())

            traj_loss = LogLoss(probs,actions,reward,baseline)
            baseline = 0.99*baseline + 0.01*reward
            w.add_scalar('Baseline',baseline,count)
            if i == 0:
                total_loss = traj_loss
            else:
                total_loss += traj_loss
        ## Averaging to create loss estimator
        total_loss = torch.mul(total_loss,1/num_trajectory)
        w.add_scalar('Loss', total_loss.data[0],episode)
        ################################################################
        if episode<150:
            optimizer = optimizer1
        elif episode<500:
            optimizer = optimizer2
            scheduler2.step()
        else:
            optimizer = optimizer3
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        # after_weights_list =layerMag(net)
        # relDiff_list = relDiff(before_weights_list,after_weights_list)
        # relDiff_dict = listToDict(relDiff_list)
        # w.add_scalars('LayerChanges',relDiff_dict,count)
        # weight_change = totalDiff(before_weights_list,after_weights_list)
        # w.add_scalar('Weight Change',weight_change,count)
        after_weights = netMag(net)
        w.add_scalar('Weight Change', abs(before_weights-after_weights),count)
    w.close()
    return net
################################################################

neuron_parameters = [20,30,40,50]
probability_parameters = [0.2,0.3,0.4,0.5]
min_runs = 500
run_count =0
for prob in probability_parameters:
    for neuron in neuron_parameters:
        run_count +=1
        print("Run {}",run_count)
        model = trainModel(prob, neuron)
        # average_runs = evaluateModel(model)
        average_runs, std = averageModelRuns(model)
        print("Hidden Units: {}, Dropout Prob: {}".format(neuron,prob))
        print("Mean runs: {}, Standard Deviation: {}".format(average_runs,std))
        ipdb.set_trace()
        if average_runs<min_runs:
            best_model = model
            min_runs = average_runs

torch.save(best_model.state_dict(),'best_model.pt')
