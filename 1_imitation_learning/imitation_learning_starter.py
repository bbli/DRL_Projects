#==========================================================================  
# Imitation Learning Starter Code 
# 
# The code was initially written for UCSB Deep Reinforcement Learning Seminar 2018
#
# Authors: Jieliang (Rodger) Luo, Sam Green
#
# April 9th, 2018
#==========================================================================

import gym
# import necessary libraries here
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from time import sleep
import numpy as np
import ipdb
from tensorboardX import SummaryWriter

from utils import *
from functions import *

num_episodes = 3

# build neural network here 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.layer1 = nn.Linear(2,5)
        # self.layer2 = nn.Linear(5,5)
        # self.layer3 = nn.Linear(5,5)
        # self.layer4 = nn.Linear(5,3)

        self.fc1 = nn.Linear(2,10)
        # self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,3)
    def forward(self,x):
        # x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        # x = F.relu(self.layer4(x))

        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.training:
            return F.softmax(x,dim=1)
        else:
            return F.softmax(x,dim=0)
    def weightInitialization(self):
        for i,params in enumerate(self.parameters()):
            if (i%2==0):
                params = nn.init.normal(params,0,1)



if __name__ == "__main__":
    # create environment
    env = gym.make('MountainCar-v0')

    print("hi")
    print(env.action_space) # 3 actions: push left, no push, push right
    print(env.observation_space) # 2 observations: position, velocity 
    print(env.observation_space.high) # max position & velocity: 0.6, 0.07
    print(env.observation_space.low) # min position & velocity: -1.2, -0.07
    print("hello")

    ################### **Initializing Network Objects** #########################
    net = Net().double()
    net.weightInitialization()
    net.train()
    # optimizer = optim.Adam(net.parameters(),lr=9e-1, weight_decay =0)
    criterion = nn.CrossEntropyLoss()
    lr_generator = LRGenerator()
    
    count =0
    with SummaryWriter('L/run2') as w:
        w.add_text('Parameter',"this is a 1 hidden layer neural net")
        w.add_text('Parameters',str(net))
        for episode in range(num_episodes):
                    
                ################### **Getting and formatting the human trained data** #########################
                
                observations, actions = dataCollector(env)
                observations, actions = np.array(observations), np.array(actions)
                # pdb.set_trace()
                observations,actions = balanceDataset(observations,actions)
                # scalar = Standarize()
                # observations = scalar(observations)
                # print("Observation mean: {}".format(observations.mean()))
                # print("Observation std: {}".format(observations.std()))
                observations, actions = torch.DoubleTensor(observations), torch.LongTensor(actions)
                observations, actions = tensor_format(observations), tensor_format(actions)

                ################### **Training the Network** #########################
                learn_rate = next(lr_generator)
                optimizer = optim.SGD(net.parameters(),lr=learn_rate,momentum=0.90, nesterov=True,weight_decay=1e-4)
                printModel(net,optimizer)

                train_iterations=3
                for i in range(train_iterations):
                    count += 1
                    ipdb.set_trace()
                    before_weights = weightMag(net)
                    ################################################################
                    optimizer.zero_grad()
                    #########################
                    outputs = net(observations)

                    acc = score(outputs,actions)
                    w.add_scalar('Accuracy', acc,count)
                    print("Accuracy: {}".format(acc))
                    #########################
                    loss = criterion(outputs, actions)

                    w.add_scalar('Loss', loss.data[0],count)
                    print("Loss value: {}".format(loss))
                    #########################
                    loss.backward()
                    optimizer.step()
                    ################################################################
                    after_weights =weightMag(net)
                    relDiff_list = relDiff(before_weights,after_weights)
                    relDiff_dict = listToDict(relDiff_list)
                    w.add_scalars('LayerChanges',relDiff_dict,count)

                print("Network updated!")

    ################### **Evaluating Network** #########################
    
    observation = env.reset()
    done = False
    step = 0
    test_actions = []
    test_observations = [] 
    net.eval()

    while not done:
        env.render()
        # print(observation)
        
        x = Variable(torch.from_numpy(observation),requires_grad=True).double()
        out = net(x).data.numpy()
        action = np.random.choice(3,1,p=out)
        action =int(action)
        # print("Current step: {}".format(step))
        print(action)
        observation, reward, done, info = env.step(action)

        # store all the observations and actions from one episode
        test_observations.append(observation)
        test_actions.append(action)
        
        step += 1
        sleep(0.05)

