import gym
import numpy as np
import random
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import ipdb
from utils import *
from functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import torch.nn.init as init



# build a one-layer network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(16,4,bias=False)
        print(next(self.parameters()))
        for param in self.parameters():
            init.constant(param,0)
        print(next(self.parameters()))
    def forward(self, x):
        return self.linear(x)

def weightInitialization(m):
    if isinstance(m,nn.Linear):
        m.weight.data.normal_(0, 0.25*math.sqrt(1. / 20))
        print("Changed {}!".format(m.__class__.__name__))



env = gym.make('FrozenLake-v0')

print(env.action_space)
print(env.observation_space) # discrete finite observations

q_function = Net()
# q_function.apply(weightInitialization)
criterion = nn.MSELoss()
count =0
random_count =0
num_episodes = 2000
max_steps = 99
w = SummaryWriter()
observations_list=[]
actions_list=[]
rewards_list =[]
random_q_table = np.zeros([env.observation_space.n,env.action_space.n])
count =0
for episode in range(num_episodes):
    
    # Reset environment and get first new observation
    observation = env.reset()
    rewards = 0
    done = False
    step = 0
    random_episode =500
    # if episode==random_episode:
        # print("got Qtable")
        # random_q_table = createQTable(q_function)
    
    # epsilon = epsilonChooser(episode)
    epsilon = 0.4# e-greedy policy 
    learn_rate = LRGenerator(episode)
    optimizer = torch.optim.SGD(q_function.parameters(),lr=learn_rate)
    #The Q-Network
    while step < max_steps:
        count +=1             
        step += 1

        # the following two lines should be commented out during the training to speed the learning process
        # env.render()
        # time.sleep(1)

        ## We slice to create a 2d matrix rather than just indexing,
        ## which would return a vector
        # observation_state = np.identity(16)[observation:observation+1]
        ################### **Get Action** #########################
        observation_state = oneHotState(observation)
        # observation_state = tensorFormat(torch.FloatTensor(observation_state))
        if (np.random.random()<epsilon) and (episode<random_episode):
            action = env.action_space.sample()
            random_count +=1
            getStateandAction(observation,action,observations_list,actions_list)
        else:
            # action = getAction(observation_state,q_function)
            action = getAction2(observation_state, q_function)
        ################################################################

        observation_new, reward, done, info = env.step(action)
        new_observation_state = oneHotState(observation_new)
        
        # target_q_value = getTarget(new_observation_state,q_function,reward)
        target_q_value = getTarget2(new_observation_state,q_function,reward)
        # ipdb.set_trace()
        ################### **Prepping Data Model** #########################

        target_q_value = tensorFormat(torch.FloatTensor(target_q_value))

        ################### **Updating Model** #########################
        count += 1
        # ipdb.set_trace()
        # before_weights = weightMag(q_function)
        #########################
        optimizer.zero_grad()
        observation_state = tensorFormat(torch.FloatTensor(observation_state)) 
        outputs = q_function(observation_state)
        output = outputs[0,action]
        loss = criterion(output,target_q_value)

        w.add_scalar('Loss', loss.data[0],count)
        # print("Loss value: {}".format(loss.data[0]))


        loss.backward()
        optimizer.step()
        #########################
        # after_weights =weightMag(q_function)
        # relDiff_list = relDiff(before_weights,after_weights)
        # relDiff_dict = listToDict(relDiff_list)
        # w.add_scalars('LayerChanges',relDiff_dict,count)

        ################### **Ending Stuff** #########################
        
        rewards += reward
        observation = observation_new
        actions_list.append(int(action))
        observations_list.append(int(observation))
        w.add_scalar('Observations',observation,count)
        w.add_scalar('Action',action,count)
        # w.add_histogram('Observations Distribution',observation)
        
        if done == True:
            #Reduce chance of random action as we train the model.
            # print("episode {} ends".format(episode))
            print("rewards are {}".format(rewards))
            break
    rewards_list.append(rewards)
    w.add_scalar('Reward',rewards,episode)
# print("Percentage of random actions: {}".format(random_count/(total_step)))
# w.add_histogram('Observations Distribution',np.array([observations_list]))
w.close()
print(next(q_function.parameters()))

sa_dict = {'state':observations_list,'action':actions_list}
sa_df = pd.DataFrame.from_dict(sa_dict)
# sns.regplot(x='state', y= 'action',data = sa_df)
sns.jointplot(x='state', y= 'action',data = sa_df)
plt.show()

# ################### **Evaluation** #########################
# for episode in range(num_episodes):
    # observation = env.reset()
    # rewards = 0
    # done = False
    # step = 0
    # while step < max_steps:
        # step += 1

    # # 1. use neural net to get action
    # # 2. feed action into env
    # # 3. feed new state back into neural net

    # observation_state = oneHotState(observation)
    # action = getAction(observation_state,q_function)
    # observation_new, reward, done, info = env.step(action)

    # rewards += reward
    # observation = observation_new

    # if done == True:
        # #Reduce chance of random action as we train the model.
        # # print("episode {} ends".format(episode))
        # print("rewards are {}".format(rewards))

        # break
