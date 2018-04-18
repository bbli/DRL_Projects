import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils import *

# # test = torch.Tensor(3,2)
# # print(test)
# # print(torch.max(test,0))
# # print(torch.max(test,1))

# class Net(nn.Module):
    # def __init__(self):
        # super().__init__()
        # self.layer1 = nn.Linear(2,5)
        # self.layer2 = nn.Linear(5,5)
        # self.layer3 = nn.Linear(5,5)
        # self.layer4 = nn.Linear(5,3)
    # def forward(self,x):
        # x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        # x = F.relu(self.layer4(x))
        # return F.softmax(x,dim=0)


# tensor_list=[]
# net = Net()

# dummy = torch.Tensor(2)
# dummy = Variable(dummy,requires_grad=False)

# out = net(dummy)

# x,y = torch.max(out,0)

# for i,params in enumerate(net.parameters()):
    # print(i)
    # print(params)

################### **Figuring out reasonable inital loss value** #########################

# class Net(nn.Module):
    # def __init__(self):
        # super().__init__()
        # self.layer1 = nn.Linear(2,5)
        # self.layer2 = nn.Linear(5,5)
        # self.layer3 = nn.Linear(5,5)
        # self.layer4 = nn.Linear(5,3)
    # def forward(self,x):
        # x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        # x = F.relu(self.layer4(x))
        # return F.log_softmax(x,dim=1)
        # # return x

# criterion = nn.NLLLoss()
# # criterion = nn.CrossEntropyLoss()
# net = Net()

# # samples = 8
# # dummy_input = torch.ones(samples,2)
# # label = torch.zeros(samples)
# # dummy_input, label = tensor_format(dummy_input), tensor_format(label).long()

# # pred = net(dummy_input)
# # loss = criterion(pred, label)
# # print(loss.data[0])
# # print("Loss value: {}".format(loss))

# ################### **Testing variable scope** #########################
# import getch
# def humanInput():
    # invalid = True
    # while invalid:
        # char = getch.getch()
        # # char = char.decode("utf-8") # you need this line if you are running Windows 
                                
        # if char == 'a':
            # a = 0
            # break
        # elif char == 's':
            # a = 1
            # break
        # elif char == 'd':
            # a = 2
            # break
        # elif char == 'b':
            # invalid = False
    # return a
# ################### **Weight initialization** #########################
# # for params in net.parameters():
    # # print(params.shape)
    # # mean = float(params.mean().data.numpy())
    # # print(mean)
# for i,params in enumerate(net.parameters()):
    # print("{} layer".format(i))
    # print(params)

# ################### **Data Standarization** #########################

# ################### **Learning how to Implement my Own Loss Function** #########################
# from torch.autograd import Function

# class LinearFunction(Function):
    # def forward(ctx,input, weight, bias = None)

# optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.90, nesterov=True,weight_decay=1e-4)
# for weight in optimizer.param_groups[0]['params']
    # print

# d_p =optimizer.param_groups[0]['params'][0].data

# type(optimizer.param_groups[0]['params'])

# params = optimizer.param_groups[0]['params']

# first = params[0]

# first.grad
# ################### **Getting ipython to work with gym** #########################
# import gym
# env = gym.make('CartPole-v0')
# env.reset()
# env.render()

################### **Getting tensorboard to log text** #########################
from tensorboardX import SummaryWriter

writer = SummaryWriter('Test')
writer.add_text("Text","hi")
writer.close()
# Yep, doesn't work. Either my pytorch version is too new or my tensorflow is not working as expected

