from tensorboardX import SummaryWriter
import os

from Experiment import *

Pendulum = ExperimentClass('Pendulum-v0')
# os.chdir("debug")
os.chdir("trainModel_runs")
# actor_neuron_parameters = [25,35,45]
# critic_neuron_parameters = [4,5,6]
# min_reward = -100
# for actor_neuron in actor_neuron_parameters:
    # for critic_neuron in critic_neuron_parameters:
        # w = SummaryWriter()
        # model = Lunar.trainModel(actor_neuron,critic_neuron,w)
        # average_reward,std = Lunar.averageModelRuns(model,w)
        # w.close()
        # print("Actor Hidden Units: {} Critic Hidden Units: {}".format(actor_neuron,critic_neuron))
        # print("Mean rewards: {}, Standard Deviation: {}".format(average_reward,std))
        # if average_reward > min_reward:
            # best_model = model
            # min_reward = average_reward
w = SummaryWriter()
target_actor_model = Pendulum.trainModel(36,6,w)
w.close()
