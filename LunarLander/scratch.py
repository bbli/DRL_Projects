import ipdb
def trainModel(self,neurons,w):
    ################ **Defining Model and Environment** ##################
    env = gym.make(self.environment)
    net = Net(neurons)
    value_net = Net(10)
    ## adding a pointer to the net
    self.current_model = net
    ################ **Experiment Hyperparameters** ##################
    num_episodes = 1000
    ## figured this out experimentally
    baseline = -160
    num_trajectory = 5
    lr_1 = 0.01
    optimizer1 = optim.Adam(net.parameters(), lr=lr_1)


    w.add_text("Experiment Parameters","Hidden Units: {} Number of episodes: {} Trajectory Size: {} Adam Learning Rate 1: {} ".format(neurons,num_episodes,num_trajectory,lr_1))
    ################################################################
    count = 0
    for episode in range(num_episodes):
        before_weights = netMag(net)
        ################# **Training** ###################
        # total_loss, count, baseline = getTrajectoryLoss(net,env,count,baseline,episode,w)
        # total_loss, count, baseline = getTotalLoss(net,env,count,baseline,episode,num_trajectory,w)

        traj_s_a_list, traj_nodes_list = getSamples()
        ipdb.set_trace()
        value_net = fitValueFunction(traj_s_a_list,value_net)
        getTotalLoss(nodes_list,rewards_list,states_list,value_net)
        updateNetwork(optimizer1,total_loss)
        ################################################################
        after_weights = netMag(net)
        w.add_scalar('Weight Change', abs(before_weights-after_weights),count)
    return net
