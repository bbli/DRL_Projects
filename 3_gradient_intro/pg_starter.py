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
from Environment import *


@timeit
def trainModel(environment,neurons):
    ################ **Defining Model and Environment** ##################
    env = gym.make(environment.environment)
    net = generateNetwork(env,neurons)
    w = SummaryWriter()
    # print(env.action_space)
    # print(env.observation_space)
    # showModel(net)
    # randomWalk()

    ################################################################
    count = 0
    num_episodes = 1000
    baseline = -500
    num_trajectory = 10
    lr_1 = 0.01
    # lr_2 = 4e-3
    optimizer1 = optim.Adam(net.parameters(), lr=lr_1)
    # optimizer2 = optim.Adam(net.parameters(),  lr=lr_2)
    w.add_text("Experiment Parameters","Hidden Units: {} Number of episodes: {} Trajectory Size: {} Adam Learning Rate 1: {} ".format(neurons,num_episodes,num_trajectory,lr_1))
    # optimizer2 = optim.SGD(net.parameters(),  lr=lr_2,momentum=0.8, nesterov = True)
    # scheduler2 = LambdaLR(optimizer2,lr_lambda=cyclic(210))
    for episode in range(num_episodes):
        # print(episode)
        # before_weights_list = layerMag(net)
        before_weights = netMag(net)
        ################# **Evaluating the Loss across Trajectories** ###################
        # total_loss, count, baseline = getTrajectoryLoss(net,env,count,baseline,episode,w)
        total_loss, count, baseline = getTotalLoss(net,env,count,baseline,episode,num_trajectory,w)
        ################################################################
        # if episode<300:
        updateNetwork(optimizer1,total_loss)
        # else:
            # updateNetwork(optimizer2,total_loss,scheduler2)


        after_weights = netMag(net)
        w.add_scalar('Weight Change', abs(before_weights-after_weights),count)
    w.close()
    return net
################################################################

# neuron_parameters = [16,18,20,22]
# x,y = len(neuron_parameters), len(neuron_parameters)
# average_run_table = np.zeros((x,y))
# std_table = np.zeros((x,y))

# min_runs = 500
# run_count =0
# for j,neuron in enumerate(neuron_parameters):
    # run_count +=1
    # print("Run {}",run_count)
    # model = trainModel(neuron)
    # # average_runs = evaluateModel(model)
    # average_runs, std = averageModelRuns(model)
    # print("Hidden Units: {}".format(neuron))
    # print("Mean runs: {}, Standard Deviation: {}".format(average_runs,std))
    # average_run_table[i,j] = average_runs
    # std_table[i,j] = std
    # if average_runs<min_runs:
        # best_model = model
        # min_runs = average_runs
environment = EnvironmentClass('Acrobot-v1')
neuron_parameters = [16,18,20,22]
average_run_table = []
std_table = []

min_runs = 500
run_count =0
for j,neuron in enumerate(neuron_parameters):
    run_count +=1
    print("Run {}",run_count)
    model = trainModel(environment,neuron)
    # average_runs = evaluateModel(model)
    average_runs, std = environment.averageModelRuns(model)
    print("Hidden Units: {}".format(neuron))
    print("Mean runs: {}, Standard Deviation: {}".format(average_runs,std))
    average_run_table.append(average_runs)
    std_table.append(std_table)
    if average_runs<min_runs:
        best_model = model
        min_runs = average_runs

# torch.save(best_model.state_dict(),'best_baseline_tuning_model.pt')
