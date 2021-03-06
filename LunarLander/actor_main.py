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
        self.fc2 = nn.Linear(neurons,neurons)
        self.final = nn.Linear(neurons,1)
    def forward(self,x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = self.final(x)
        return x

class CriticClass():
    def __init__(self,neurons):
        self.CriticNet = CriticNet(neurons)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.CriticNet.parameters(),lr = 6e-3)
        self.count = 0
    def fit(self,states,targets,w):
        self.count += 1
        targets = numpyFormat(targets).float()
        states = numpyFormat(states).float()
        pred = self.CriticNet.forward(states)

        loss = self.criterion(pred,targets)
        w.add_scalar("Critic Loss",loss.data[0],self.count)

        updateNetwork(self.optimizer,loss) 


class Experiment(EnvironmentClass):
    def __init__(self,string):
        self.environment = string
        self.current_model = None
        
        self.runs_test_rewards_list = []
        self.runs_models_list = []
    def episodeLogger(self,episode):
        self.episode = episode

    @timeit
    def trainModel(self,actor_neurons,critic_neurons,w):
        ################ **Defining Model and Environment** ##################
        env = gym.make(self.environment)
        actor_net = ActorNet(actor_neurons)
        Critic = CriticClass(critic_neurons)
        ## adding a pointer to the net
        self.current_model = actor_net
        ################ **Experiment Hyperparameters** ##################
        num_episodes = 1000
        ## figured this out experimentally
        baseline = -240
        num_trajectory = 8
        epsilon = 1e-8
        lr_1 = 4e-3
        optimizer = optim.Adam(actor_net.parameters(), lr=lr_1, eps=epsilon)


        w.add_text("Experiment Parameters","ActorNet Hidden Units: {} CriticNet Hidden Units: {} Number of episodes: {} Trajectory Size: {} Adam Learning Rate 1: {} ".format(actor_neurons,critic_neurons,num_episodes,num_trajectory,lr_1))
        ################################################################ count = 0
        for episode in range(num_episodes):
            self.episodeLogger(episode)
            episodePrinter(episode,400)
            before_weights = netMag(actor_net)
            ################# **Training** ###################
            traj_s_r_list, traj_nodes_list = getSamples(actor_net,env,num_trajectory)
            # states1, targets1 = createStatesAndTargets(traj_s_r_list,Critic.CriticNet)
            states,targets = createStatesAndMCTargets(traj_s_r_list)
            Critic.fit(states,targets,w)

            advantage_list = createAdvantage(traj_s_r_list,Critic.CriticNet)
            total_loss = getBootstrappedAdvantageLogLoss(traj_nodes_list,advantage_list)
            total_loss = torch.mul(total_loss,1/num_trajectory)

            updateNetwork(optimizer,total_loss)
            ################# **Logging** ###################
            w.add_scalar("Advantage",advantage_list.mean(),episode)
            w.add_scalar('Loss', total_loss.data[0],episode)

            mean_last_advantage = getMeanLastAdvantage(traj_s_r_list,Critic.CriticNet)
            w.add_scalar('Advantage Last',mean_last_advantage,episode)

            mean_reward = getMeanTotalReward(traj_s_r_list)
            w.add_scalar('Mean Reward',mean_reward,episode)

            avg_lr = averageAdamLearningRate(optimizer,epsilon,lr_1)
            w.add_scalar('Learning Rate',avg_lr,episode)

            after_weights = netMag(actor_net)
            w.add_scalar('Weight Change', abs(before_weights-after_weights),episode)
        return actor_net

Lunar = Experiment('LunarLander-v2')
# os.chdir("debug")
# os.chdir("trainModel_runs")
actor_neuron_parameters = [25,35,45]
critic_neuron_parameters = [4,5,6]
min_reward = -100
for actor_neuron in actor_neuron_parameters:
    for critic_neuron in critic_neuron_parameters:
        w = SummaryWriter()
        model = Lunar.trainModel(actor_neuron,critic_neuron,w)
        average_reward,std = Lunar.averageModelRuns(model,w)
        w.close()
        print("Actor Hidden Units: {} Critic Hidden Units: {}".format(actor_neuron,critic_neuron))
        print("Mean rewards: {}, Standard Deviation: {}".format(average_reward,std))
        if average_reward > min_reward:
            best_model = model
            min_reward = average_reward
