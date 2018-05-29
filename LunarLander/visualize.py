import torch.nn as nn
import torch
import torch.nn.functional as F
import ipdb

from functions import *
from utils import *
from Environment import *

class Net(nn.Module):
    def __init__(self,neurons):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8,neurons)
        # self.fc2 = nn.Linear(neurons,neurons)
        self.final = nn.Linear(neurons,4)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.final(x)
        if len(x.shape)==1:
            return F.softmax(x,dim=0)
        else:
            return F.softmax(x,dim=1)
class Experiment(EnvironmentClass):
    def __init__(self,string):
        self.environment = string
        self.current_model = None
        self.optimizer = None
        self.runs_test_rewards_list = []
        self.runs_models_list = []
    def episodeLogger(self,episode):
        self.episode = episode


environment = Experiment('LunarLander-v2')
environment.tryEnvironment()

first_model = Net(45)
first_model.load_state_dict(torch.load('onelayer_baseline_model.pt'))

print(environment.averageModelRuns(first_model))
environment.showModel(first_model)

