import ipdb
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
        self.optimizer = optim.Adam(self.CriticNet.parameters(),lr = 5e-4)
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


class ExperimentClass(EnvironmentClass):
    def __init__(self,string):
        super().__init__(string)
        self.current_actor_net = None
        self.current_critic_net = None
        
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
        self.current_actor_net = target_actor_net
        self.current_critic_net = target_critic_net
        ################ **Experiment Hyperparameters** ##################
        num_episodes = 1000
        max_steps = 100
        memory_threshold=1000
        memory_buffer = []
        N = 100
        epsilon = 1e-8
        lr_1 = 2e-2
        optimizer = optim.Adam(actor_net.parameters(), lr=lr_1, eps=epsilon)


        w.add_text("Environment Parameters","Number of Episodes: {} Max Steps: {}".format(num_episodes,max_steps))
        w.add_text("Model Parameters","ActorNet Hidden Units: {} CriticNet Hidden Units: {} Adam Learning Rate: {} Memory Size: {} Memory Batch Size: {}".format(actor_neurons,critic_neurons,lr_1,memory_threshold,N))
        ################################################################ 
        count = 0
        for episode in range(num_episodes):
            ## 3
            self.episodeLogger(episode)
            episodePrinter(episode,400)
            ################################################################
            state = env.reset()
            total_reward = 0
            for steps in range(max_steps):
                count +=1
                ################# **Sampling** ###################
                action = getContinuousAction(actor_net,state)
                new_state, reward, done, info = env.step(action)
                total_reward += reward
                memory_buffer.append((state,action,reward,new_state))
                if len(memory_buffer)<memory_threshold:
                    pass
                else:
                    memory_buffer.pop(0)
                    
                    states,actions,rewards,new_states = getMiniBatch(memory_buffer,N)
                    ################ **Critting Fitting** ##################

                    gamma = 0.95
                    new_states_actions = createActionNodes(new_states,target_actor_net).data.numpy()
                    new_states_q_values = createQValues(new_states,new_states_actions,target_critic_net)

                    targets = rewards + gamma*new_states_q_values
                    Critic.fit(states,actions,targets,w)

                    before_weights = netMag(actor_net)
                    # t_before_critic_weights = netMag(Critic.CriticNet)
                    ################ **Updating Policy** ##################
                    optimal_action_nodes = createActionNodes(states,actor_net)
                    optimal_q_values = createQValueNodes(states,optimal_action_nodes,Critic.CriticNet)
                    loss = optimal_q_values.mean()
                    loss = torch.mul(loss,-1)

                    updateNetwork(optimizer,loss)
                    ################ **Updating Target Networks** ##################
                    updateTargetNetwork(Critic.CriticNet,target_critic_net)
                    updateTargetNetwork(actor_net,target_actor_net)
                    
                    ################# **Logging** ###################
                    ## 5:once memory_buffer is large enough
                    # t_after_critic_weights = netMag(Critic.CriticNet)
                    # t_diff = t_after_critic_weights-t_before_critic_weights


                    w.add_scalar('Loss', loss.data[0],count)

                    avg_lr = averageAdamLearningRate(optimizer,epsilon,lr_1)
                    w.add_scalar('Learning Rate',avg_lr,count)

                    after_weights = netMag(actor_net)
                    w.add_scalar('Weight Change', abs(before_weights-after_weights),count)
                ## 4:inside max step for loop
                state = new_state
            ## 3: inside episode for loop
            w.add_scalar('Reward',total_reward,episode)
        return target_actor_net

