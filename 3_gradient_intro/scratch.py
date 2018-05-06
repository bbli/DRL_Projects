import torch.nn as nn
import torch.nn.functional as F

from functions import *
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6,16)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(16,3)
    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        print(x)
        print("Shape: {}".format(x.shape))
        return F.softmax(x)

net = Net()
net.load_state_dict(torch.load('May04-19:11.pt'))

# count = evaluateModel(net)
# print(count)
# print(averageModelRuns(net))

data = torch.ones(5,6)
data = Variable(data).float()
