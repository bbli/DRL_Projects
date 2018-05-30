import gym
import torch.nn as nn
import numpy as np
from Environment import *

# class ActorNet(nn.Module):
    # def __init__(self,neurons):
        # super().__init__()
        # self.fc1 = nn.Linear(8,neurons)
        # # self.fc2 = nn.Linear(neurons,neurons)
        # self.final = nn.Linear(neurons,4)
    # def forward(self,x):
        # x = F.relu(self.fc1(x)) 
        # # x = F.relu(self.fc2(x))
        # x = self.final(x)
        # if len(x.shape)==1:
            # return F.softmax(x,dim=0)
        # else:
            # return F.softmax(x,dim=1)

# class CriticNet(nn.Module):
    # def __init__(self,neurons):
        # super().__init__()
        # self.fc1 = nn.Linear(8,neurons)
        # self.fc2 = nn.Linear(neurons,neurons)
        # self.final = nn.Linear(neurons,1)
    # def forward(self,x):
        # x = F.relu(self.fc1(x)) 
        # x = F.relu(self.fc2(x))
        # x = self.final(x)
        # return x

# class CriticClass():
    # def __init__(self,neurons):
        # self.CriticNet = CriticNet(neurons)
        # self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.CriticNet.parameters(),lr = 6e-3)
        # self.count = 0
    # def fit(self,states,targets,w):
        # self.count += 1
        # targets = numpyFormat(targets).float()
        # states = numpyFormat(states).float()
        # pred = self.CriticNet.forward(states)

        # loss = self.criterion(pred,targets)
        # w.add_scalar("Critic Loss",loss.data[0],self.count)

        # updateNetwork(self.optimizer,loss) 
# actor_net = ActorNet(2)
# target_actor_net = ActorNet(2)
# target_actor_net.load_state_dict(actor_net.state_dict())


class ActorNet(nn.Module):
    def __init__(self,neurons):
        super().__init__()
        self.fc1 = nn.Linear(8,neurons)
        # self.fc2 = nn.Linear(neurons,neurons)
        self.final = nn.Linear(neurons,1)
    def forward(self,x):
        x = F.relu(self.fc1(x)) 
        # x = F.relu(self.fc2(x))
        x = self.final(x)
        return 2*F.tanh(x)
Net = ActorNet(10)
state = numpyFormat(np.ones(8)).float()
output = Net(state)
