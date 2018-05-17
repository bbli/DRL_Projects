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
        self.fc1 = nn.Linear(6,neurons)
        # self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(neurons,3)
    def forward(self, x):
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        # print(x)
        # print("Shape: {}".format(x.shape))
        return F.softmax(x)

environment = Environment('Acrobot-v1')
environment.tryEnvironment()
first_model = Net(16)
first_model.load_state_dict(torch.load('models/May04-19:11.pt'))

print(environment.averageModelRuns(first_model))

print(averageModelRuns(first_model))
# environment.showModel(first_model)

# best_model = Net(18)
# best_model.load_state_dict(torch.load('models/lowest_std.pt'))
# environment.showModel(best_model)

