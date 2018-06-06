import numpy as np
import torch
from utils import *
import ipdb

# def getAction(observation_vector,q_function):
    # action_values = q_function(observation_vector)
    # return np.argmax(action_values.data.numpy())

################################################################
def getAction2(observation_vector,q_function):
   observation_vector = tensorFormat(torch.FloatTensor(observation_vector)) 
   q_values = q_function(observation_vector)
   q_values = q_values.data.numpy()
   return int(np.argmax(q_values))

def getTarget2(new_observation_state,q_function,reward):
    gamma = 0.9

    observation_vector = tensorFormat(torch.FloatTensor(new_observation_state)) 
    q_values = q_function(observation_vector)
    q_values = q_values.data.numpy()

    max_q_value = float(np.max(q_values))

    return np.array([reward + gamma*max_q_value]).reshape(1,1)

def getAction(observation_vector, q_function):
    q_values = getQValues(observation_vector,q_function)
    return np.argmax(q_values)

def getTarget(new_observation_state,q_function,reward):
    gamma =0.9
    q_values = getQValues(new_observation_state,q_function)
    max_q_value = np.max(q_values)

    ## reshape to make into 2d matrix, even though we only feed one sample at a time
    return np.array([reward+gamma*max_q_value]).reshape(1,1)



def getQValues(observation_vector,q_function):
    q_value_list =[]
    possible_actions = np.arange(4)
    for action in possible_actions:
        action_state = oneHotAction(action)
        product_state = np.concatenate((observation_vector,action_state),axis=1)
        product_state = tensorFormat(torch.FloatTensor(product_state))
        q_value = q_function(product_state)
        q_value = float(q_value.data.numpy())
        q_value_list.append(q_value) 
    return np.array(q_value_list)
################################################################

def alphaChooser(episode):
    # assert episode >= 0, "Not a valid episode"
    # starting_epsilon = 0.5
    # if episode<1000:
        # return starting_epsilon
    # else:
        # return 0.1
    if episode<400:
        return 0.2
    elif episode<800:
        return 0.2
    elif episode<1200:
        return 0.1
    else:
        return 0.1

# def getRandomQTable(episode,random_episode,q_table,random_q_table):
    # if episode==random_episode:
        # print("got Qtable")
        # global random_q_table = np.copy(q_table)
        # ipdb.set_trace()

def getStateandAction(observation,action,observations_list,actions_list):
    # we can reference global variable w/o using global keyword because the changes are not assigments
    observations_list.append(observation)
    actions_list.append(action)

def QTableChecker(q_table,episode):
    q_table_max = np.max(q_table)
    if q_table_max>10:
        print("Overloaded at episode: {}".format(episode))
        raise Exception

def createQTable(q_function):
    table=[]
    for i in range(16):
        observation_state = oneHotState(i)
        row = []
        for j in range(4):
            action_state = oneHotAction(j)
            product_state = np.concatenate((action_state,observation_state),axis=1)
            product_state = tensorFormat(torch.FloatTensor(product_state))
            value = q_function(product_state)
            # ipdb.set_trace()
            row.append(float(value.data.numpy()))
        table.append(row)
    return np.array(table)

