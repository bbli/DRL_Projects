import torch.nn as nn
import torch.nn.functional as F

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
net.load_state_dict(torch.load('May04-19:11.pt'))

count,state = evaluateModel(net)
print(count)
print(state)
