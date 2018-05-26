import ipdb
import os
from tensorboardX import SummaryWriter
from Environment import *
class ActorNet(nn.Module):
    def __init__(self,neurons):
        super().__init__()
        self.fc1 = nn.Linear(8,neurons)
        # self.fc2 = nn.Linear(neurons,neurons)
        self.final = nn.Linear(neurons,4)
    def forward(self,x):
        x = F.relu(self.fc1(x)) 
        # x = F.relu(self.fc2(x))
        x = self.final(x)
        if len(x.shape)==1:
            return F.softmax(x,dim=0)
        else:
            return F.softmax(x,dim=1)

class CriticNet(nn.Module):
    def __init__(self,neurons):
        super().__init__()
        self.fc1 = nn.Linear(8,neurons)
        # self.fc2 = nn.Linear(neurons,neurons)
        self.final = nn.Linear(neurons,1)
    def forward(self,x):
        x = F.relu(self.fc1(x)) 
        # x = F.relu(self.fc2(x))
        x = self.final(x)
        return x

class Critic():
    def __init__(self,neurons):
        self.CriticNet = CriticNet(neurons)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.CriticNet.parameters(),lr = 5e-3)
    def fit(self,states,targets):
        targets = numpyFormat(targets).float()
        states = numpyFormat(states).float()
        pred = self.CriticNet.forward(states)

        loss = self.criterion(pred,targets)

        updateNetwork(self.optimizer,loss) 


class Experiment(EnvironmentClass):
    def __init__(self,string):
        self.environment = string
        self.current_actor = None
        self.current_critic = None
        self.optimizer = None
        self.runs_test_rewards_list = []
        self.runs_models_list = []
    def episodeLogger(self,episode):
        self.episode = episode

    @timeit
    def trainModel(self,actor_neurons,critic_neurons,w):
        ################ **Defining Model and Environment** ##################
        env = gym.make(self.environment)
        actor_net = ActorNet(actor_neurons)
        critic = Critic(critic_neurons)
        ## adding a pointer to the net
        self.current_actor_net = actor_net
        self.current_critic = critic
        ################ **Experiment Hyperparameters** ##################
        num_episodes = 1000
        ## figured this out experimentally
        baseline = -240
        num_trajectory = 5
        lr_1 = 0.01
        optimizer = optim.Adam(actor_net.parameters(), lr=lr_1)


        w.add_text("Experiment Parameters","ActorNet Hidden Units: {} CriticNet Hidden Units: {} Number of episodes: {} Trajectory Size: {} Adam Learning Rate 1: {} ".format(actor_neurons,critic_neurons,num_episodes,num_trajectory,lr_1))
        ################################################################ count = 0
        for episode in range(num_episodes):
            before_weights = netMag(actor_net)
            ################# **Training** ###################
            traj_s_a_list, traj_nodes_list = getSamples(actor_net,env,num_trajectory)
            states, targets = createStatesAndTargets(traj_s_a_list,critic.CriticNet)
            critic.fit(states,targets)
            advantage = createAdvantage(traj_s_a_list,critic.CriticNet)
            total_loss = getBootstrappedAdvantageLogLoss(traj_nodes_list,advantage)
            updateNetwork(optimizer,total_loss)
            ################################################################
            after_weights = netMag(actor_net)
            w.add_scalar('Weight Change', abs(before_weights-after_weights),count)
            logAdamLearningRate(optimizer,w)
        return net

Lunar = Experiment('LunarLander-v2')
os.chdir("debug")
w = SummaryWriter()
Lunar.trainModel(36,20,w)
w.close()
