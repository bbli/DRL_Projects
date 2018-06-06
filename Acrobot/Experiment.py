from Environment import *
class Experiment(EnvironmentClass):
    def __init__(self,string):
        self.environment = string
        self.current_model = None

    def trainModel(self,neurons,w):
        ################ **Defining Model and Environment** ##################
        env = gym.make(self.environment)
        net = generateNetwork(env,neurons)
        ## adding a pointer to the net
        self.current_model = net
        # print(env.action_space)
        # print(env.observation_space)
        # showModel(net)
        # randomWalk()

        ################################################################
        count = 0
        num_episodes = 1000
        baseline = -500
        num_trajectory = 10
        lr_1 = 0.01
        # lr_2 = 4e-3
        optimizer1 = optim.Adam(net.parameters(), lr=lr_1)
        # optimizer2 = optim.Adam(net.parameters(),  lr=lr_2)
        w.add_text("Experiment Parameters","Hidden Units: {} Number of episodes: {} Trajectory Size: {} Adam Learning Rate 1: {} ".format(neurons,num_episodes,num_trajectory,lr_1))
        # optimizer2 = optim.SGD(net.parameters(),  lr=lr_2,momentum=0.8, nesterov = True)
        # scheduler2 = LambdaLR(optimizer2,lr_lambda=cyclic(210))
        for episode in range(num_episodes):
            # print(episode)
            # before_weights_list = layerMag(net)
            before_weights = netMag(net)
            ################# **Evaluating the Loss across Trajectories** ###################
            # total_loss, count, baseline = getTrajectoryLoss(net,env,count,baseline,episode,w)
            total_loss, count, baseline = getTotalLoss(net,env,count,baseline,episode,num_trajectory,w)
            ################################################################
            # if episode<300:
            updateNetwork(optimizer1,total_loss)
            # else:
                # updateNetwork(optimizer2,total_loss,scheduler2)


            after_weights = netMag(net)
            w.add_scalar('Weight Change', abs(before_weights-after_weights),count)
        return net
