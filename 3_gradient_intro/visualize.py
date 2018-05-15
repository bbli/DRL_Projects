import torch.nn as nn
import torch
import torch.nn.functional as F
import ipdb

from functions import *
from utils import *

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

tryEnvironment()
first_model = Net(16)
first_model.load_state_dict(torch.load('models/May04-19:11.pt'))
showModel(net)

best_model = Net(18)
best_model.load_state_dict(torch.load('models/lowest_std.pt'))
showModel(best_model)

