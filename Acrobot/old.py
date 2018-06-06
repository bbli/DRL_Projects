import gym
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
# from tensorboardX import SummaryWriter
import ipdb

from utils import *
from functions import *

def getTrajectoryLoss(net,env,count,baseline,episode,w=None):
    num_trajectory=16
    for i in range(num_trajectory):
        count +=1
        trajectory, actions, reward = sampleTrajectory(net,env)
        probs = net(numpyFormat(trajectory).float())
        
        ################ **Zeroing Loss if Reward is worse than Baseline** ##################
        
        if abs(reward) < abs(baseline):
            traj_loss = LogLoss(probs,actions,reward,baseline)
        else:
            traj_loss = LogLoss(probs,actions,reward,reward)
        ################################################################
        if i == 0:
            total_loss = traj_loss
        else:
            total_loss += traj_loss
        ################ **Logging** ##################
        
        if w:
            w.add_scalar('Reward',reward,count)
            w.add_scalar('Baseline',baseline,count)
    ## Averaging to create loss estimator
    total_loss = torch.mul(total_loss,1/num_trajectory)
    if w:
        w.add_scalar('Loss', total_loss.data[0],episode)
    return total_loss,count,baseline

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6,20)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(20,3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x)

def trainModel(baseline_number):
    '''
    Returns a list of state_dicts for neural net
    '''
    ################ **Defining Model and Environment** ##################

    env = gym.make('Acrobot-v1')
    net = Net()
    net.load_state_dict(torch.load('best_model.pt'))
    net_list = []
    # w = SummaryWriter()
    # print(env.action_space)
    # print(env.observation_space)
    # showModel(net)
    # randomWalk()

    ################################################################
    count = 0
    num_episodes = 1000
    baseline = baseline_number
    num_trajectory = 16
    optimizer = optim.Adam(net.parameters(),  lr=3e-4)
    for episode in range(num_episodes):
        # print(episode)
        # before_weights_list = layerMag(net)
        before_weights = netMag(net)
        ################# **Evaluating the Loss across Trajectories** ###################
        total_loss, count, baseline = getTrajectoryLoss(net,env,count,baseline,episode)
        baseline = baseline_number
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        ################################################################
        if episode%100 == 0:
            net_list.append(net.state_dict())
    return net_list

def evaluateModels(net_list):
    for state in net_list:
        model = Net()
        model.load_state_dict(state)
        average_runs,std = averageModelRuns(model)
        if average_runs<min_runs:
            best_model = model
            min_runs= average_runs
    return best_model

baseline_parameters = [-90,-85,-80,-75,-72,-68,-65]
min_runs = 500
run_count =0
models_list= []
for baseline in baseline_parameters:
    run_count +=1
    print("Run {}",run_count)
    net_list = trainModel(baseline)
    model = evaluateModels(net_list)
    models_list.append(model)
    # average_runs = evaluateModel(model)
    average_runs, std = averageModelRuns(model)
    print("Hidden Units: {}, Dropout Prob: {}".format(neuron,prob))
    print("Mean runs: {}, Standard Deviation: {}".format(average_runs,std))
    if average_runs<min_runs:
        best_model = model
        min_runs = average_runs

torch.save(best_model.state_dict(),'baseline_best_model.pt')

