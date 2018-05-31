import ipdb
import os
from tensorboardX import SummaryWriter
from Environment import *

class ActorNet(nn.Module):
    def __init__(self,neurons):
        super().__init__()
        self.fc1 = nn.Linear(3,neurons)
        # self.fc2 = nn.Linear(neurons,neurons)
        self.final = nn.Linear(neurons,1)
    def forward(self,x):
        x = F.relu(self.fc1(x)) 
        # x = F.relu(self.fc2(x))
        x = self.final(x)
        return 2*F.tanh(x)

class CriticNet(nn.Module):
    def __init__(self,neurons):
        super().__init__()
        self.state_dim = 3
        self.action_dim =1
        self.fc1 = nn.Linear(self.state_dim,neurons)
        self.fc2 = nn.Linear(neurons+self.action_dim,neurons)
        self.final = nn.Linear(neurons,1)
    def forward(self,state,action):
        '''
        Assumes state and action are pytorch variables
        '''
        shape = state.shape
        x = F.relu(self.fc1(state)) 
        if len(shape)>1:
            x = torch.cat((x,action),1)
        else:
            x = torch.cat((x,action),0)
        x = F.relu(self.fc2(x))
        x = self.final(x)
        return x

class CriticClass():
    def __init__(self,neurons):
        self.CriticNet = CriticNet(neurons)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.CriticNet.parameters(),lr = 6e-3)
        self.count = 0
    def fit(self,states,actions,targets,w):
        '''
        Assumes states, actions, and targets are numpy arrays
        '''
        self.count += 1
        targets = numpyFormat(targets).float()
        actions = numpyFormat(actions).float()
        states = numpyFormat(states).float()
        pred = self.CriticNet.forward(states,actions)

        loss = self.criterion(pred,targets)
        w.add_scalar("Critic Loss",loss.data[0],self.count)

        # before_weights = netMag(self.CriticNet)
        updateNetwork(self.optimizer,loss) 
        # after_weights = netMag(self.CriticNet)
        # print("Critic Weight Change: ",str(after_weights-before_weights))


class Experiment(EnvironmentClass):
    def __init__(self,string):
        self.environment = string
        self.current_model = None
        
        self.runs_test_rewards_list = []
        self.runs_models_list = []
    def episodeLogger(self,episode):
        self.episode = episode

    # @timeit
    def trainModel(self,actor_neurons,critic_neurons,w):
        ################ **Defining Model and Environment** ##################
        env = gym.make(self.environment)
        actor_net = ActorNet(actor_neurons)
        Critic = CriticClass(critic_neurons)

        target_actor_net = ActorNet(actor_neurons)
        target_actor_net.load_state_dict(actor_net.state_dict())
        target_critic_net = CriticNet(critic_neurons)
        target_critic_net.load_state_dict(Critic.CriticNet.state_dict())


        ## adding pointer
        self.current_actor_net = actor_net
        self.current_critic = Critic
        ################ **Experiment Hyperparameters** ##################
        num_episodes = 1000
        max_steps = 500
        memory_threshold=1000
        memory_buffer = []
        gamma = 0.9
        epsilon = 1e-8
        lr_1 = 4e-3
        optimizer = optim.Adam(actor_net.parameters(), lr=lr_1, eps=epsilon)


        w.add_text("Experiment Parameters","ActorNet Hidden Units: {} CriticNet Hidden Units: {} Number of episodes: {} Adam Learning Rate 1: {} ".format(actor_neurons,critic_neurons,num_episodes,lr_1))
        ################################################################ count = 0
        for episode in range(num_episodes):
            self.episodeLogger(episode)
            episodePrinter(episode,400)
        ################################################################
            state = env.reset()
            for steps in range(max_steps):
                ################# **Sampling** ###################
                action = getContinuousAction(actor_net,state)
                new_state, reward, done, info = env.step(action)
                memory_buffer.append((state,action,reward,new_state))
                if len(memory_buffer)<memory_threshold:
                    pass
                else:
                    memory_buffer.pop(0)
                    
                    states,actions,rewards,new_states = getMiniBatch(memory_buffer)
                    ################ **Critting Fitting** ##################

                    gamma = 0.9
                    new_states_actions = createActionNodes(new_states,actor_net).data.numpy()
                    new_states_q_values = createQValues(new_states,new_states_actions,Critic.CriticNet)
                    targets = rewards + gamma*new_states_q_values
                    Critic.fit(states,actions,targets,w)

                    before_weights = netMag(actor_net)
                    t_before_critic_weights = netMag(Critic.CriticNet)
                    ################ **Updating Policy** ##################
                    optimal_action_nodes = createActionNodes(states,actor_net)
                    optimal_q_values = createQValueNodes(states,optimal_action_nodes,Critic.CriticNet)
                    loss = optimal_q_values.mean()

                    updateNetwork(optimizer,loss)
                    ################# **Logging** ###################
                    t_after_critic_weights = netMag(Critic.CriticNet)
                    t_diff = t_after_critic_weights-t_before_critic_weights
                    # w.add_scalar("Advantage",advantage_list.mean(),episode)
                    # w.add_scalar('Loss', total_loss.data[0],episode)

                    # mean_last_advantage = getMeanLastAdvantage(traj_s_r_list,Critic.CriticNet)
                    # w.add_scalar('Advantage Last',mean_last_advantage,episode)

                    # mean_reward = getMeanTotalReward(traj_s_r_list)
                    # w.add_scalar('Mean Reward',mean_reward,episode)

                    avg_lr = averageAdamLearningRate(optimizer,epsilon,lr_1)
                    w.add_scalar('Learning Rate',avg_lr,episode)

                    after_weights = netMag(actor_net)
                    w.add_scalar('Weight Change', abs(before_weights-after_weights),episode)
                    ipdb.set_trace()
                state = new_state
        return actor_net

Lunar = Experiment('Pendulum-v0')
os.chdir("debug")
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
