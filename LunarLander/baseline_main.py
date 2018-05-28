from tensorboardX import SummaryWriter
# from torch.optim.lr_scheduler import LambdaLR
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
    def __init__(self,string):
        self.environment = string
        self.current_model = None
        self.optimizer = None
        self.runs_test_rewards_list = []
        self.runs_models_list = []
    def episodeLogger(self,episode):
        self.episode = episode

    @timeit
    def trainModel(self,neurons,num_trajectory,w):
        ################ **Defining Model and Environment** ##################
        env = gym.make(self.environment)
        net = Net(neurons)
        ## adding a pointer to the net
        self.current_model = net
        ################ **Experiment Hyperparameters** ##################
        num_episodes = 1200
        ## figured this out experimentally
        baseline = -240
        # num_trajectory = 10
        lr_1 = 3e-3
        epsilon = 1e-8
        optimizer = optim.Adam(net.parameters(), lr=lr_1,eps= epsilon)
        # optimizer = optim.SGD(net.parameters(),lr=1e-2,momentum=0.8)
        self.optimizer = optimizer
        # scheduler = LambdaLR(optimizer,lr_lambda=cosine(210))


        w.add_text("Experiment Parameters","Hidden Units: {} Number of episodes: {} Trajectory Size: {} Adam Learning Rate 1: {} ".format(neurons,num_episodes,num_trajectory,lr_1))
        ################################################################
        count = 0
        for episode in range(num_episodes):
            self.episodeLogger(episode)
            episodePrinter(episode,400)
            
            before_weights = netMag(net)
            ################# **Training** ###################
            # total_loss, count, baseline = getTrajectoryLoss(net,env,count,baseline,episode,num_trajectory,w)
            total_loss, count, baseline = getTotalLoss(net,env,count,baseline,episode,num_trajectory,w)
            updateNetwork(optimizer,total_loss)
            # print("Updated Network on episode: ",episode)
            ################################################################
            avg_lr = averageAdamLearningRate(optimizer,epsilon,lr_1)
            w.add_scalar('Learning Rate',avg_lr,episode)
            after_weights = netMag(net)
            w.add_scalar('Weight Change', abs(before_weights-after_weights),episode)
        return net

Lunar = Experiment('LunarLander-v2')
# os.chdir("debug")
os.chdir("base_baseline")
neuron_parameters = [35,45]
num_trajectory_list = [6,8]
min_reward = 0
for neuron in neuron_parameters:
    for num_trajectory in num_trajectory_list:
        w = SummaryWriter()
        model = Lunar.trainModel(neuron,num_trajectory,w)
        average_reward,std = Lunar.averageModelRuns(model,w)
        w.close()
        print("Hidden Units: {} Num_Traj: {}".format(neuron,num_trajectory))
        print("Mean rewards: {}, Standard Deviation: {}".format(average_reward,std))
        if average_reward<min_reward:
            best_model = model
            min_reward = average_reward
