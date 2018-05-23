from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import ipdb
import os

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
    def __init__(self):
        self.environment = 'LunarLander-v2'
        self.current_model = None
        self.optimizer = None
    def episodeLogger(self,episode):
        self.episode = episode

    # @timeit
    def trainModel(self,neurons,w):
        ################ **Defining Model and Environment** ##################
        env = gym.make(self.environment)
        net = Net(neurons)
        ## adding a pointer to the net
        self.current_model = net
        ################ **Experiment Hyperparameters** ##################
        num_episodes = 1000
        ## figured this out experimentally
        baseline = -240
        num_trajectory = 10
        lr_1 = 5e-3
        epsilon = 1e-8
        optimizer = optim.Adam(net.parameters(), lr=lr_1,eps= epsilon)
        # optimizer = optim.SGD(net.parameters(),lr=1e-2,momentum=0.8)
        self.optimizer = optimizer
        # scheduler = LambdaLR(optimizer,lr_lambda=cosine(210))


        w.add_text("Experiment Parameters","Hidden Units: {} Number of episodes: {} Trajectory Size: {} SGD Learning Rate 1: {} ".format(neurons,num_episodes,num_trajectory,lr_1))
        ################################################################
        count = 0
        for episode in range(num_episodes):
            self.episodeLogger(episode)
            if episode%200 ==0:
                print("Reached Episode: ",episode)
            
            before_weights = netMag(net)
            ################# **Training** ###################
            # total_loss, count, baseline = getTrajectoryLoss(net,env,count,baseline,episode,num_trajectory,w)
            total_loss, count, baseline = getTotalLoss(net,env,count,baseline,episode,num_trajectory,w)
            updateNetwork(optimizer,total_loss)
            # print("Updated Network on episode: ",episode)
            ################################################################
            avg_lr = averageAdamLearningRate(optimizer,epsilon,lr_1)
            w.add_scalar('Learning Rate',avg_lr,count)
            after_weights = netMag(net)
            w.add_scalar('Weight Change', abs(before_weights-after_weights),count)
        return net

Lunar = Experiment()
os.chdir("single_run")
w = SummaryWriter()
model = Lunar.trainModel(36,w)
print(Lunar.averageModelRuns(model))
w.close()
