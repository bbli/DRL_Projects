from torch.autograd import Variable
import numpy as np

def tensor_format(tensor):
        new_tensor = Variable(tensor,requires_grad=False)
        return new_tensor

def customLogger(loss_value, loss_datapoints, epoch, string):
    print('Epoch: {} \t {} \tLoss: {:.6f}'.format(epoch, string,loss_value))
    loss_datapoints.append(loss_value)


def valScorer(test_loader,model,criterion):
    val_loss =0
    count =0
    for (img, label) in test_loader:
        img, label = tensor_format(img), tensor_format(label)
        label = label.long()

        pred = model(img)
        loss = criterion(pred, label)

        val_loss += loss.data[0]
        count +=1
    val_loss = val_loss/count
    return val_loss

def weightMag(net):
    layer_list =[]
    for param in net.parameters():
        mag = (param*param).sum().sqrt()
        mag = float(mag.data.numpy())
        layer_list.append(mag) 
    return layer_list

################################################################
def relDiff(list1,list2):
    l=[]
    for a,b in zip(list1,list2):
        l.append(relDiffHelper(a,b))
    return l

def relDiffHelper(a,b):
    return abs(a-b)/abs(a)
################################################################

def listToDict(l):
    new_dict ={}
    for i,item in enumerate(l):
        new_dict[str(i)]=item
    return new_dict

from sklearn.preprocessing import StandardScaler

        
class Standarize():
    def __init__(self):
        self.scalar = StandardScaler()
    def __call__(self, image):
        self.scalar.fit(image)
        return self.scalar.transform(image)

def printModel(net,optimizer):
    print(net)
    print(optimizer.defaults)

################################################################
def argMax(outputs):
    # dim=1 since output is a matrix
    # Choosing the second element to get the arg_max
    return outputs.data.max(1)[1]
def score(outputs, labels):
    pred = argMax(outputs)
    truth = (pred == labels.data)
    return truth.sum()/len(truth)

################################################################

def LRGenerator():
    yield 9e-1
    yield 3e-1
    while True:
        yield 1e-1

################################################################
def augmentData(diff, index_array, observations, actions):
    additonal_indexes = np.random.choice(index_array,diff)
    return observations[additonal_indexes], actions[additonal_indexes]


def balanceDataset(observations,actions):
    ## observations is a N by 3 numpy array and actions is a N by 1 numpy array
    right_indices = np.nonzero(actions)[0]
    left_indices = np.where(actions == 0)[0]
    if len(left_indices)<len(right_indices):
        diff = len(right_indices)-len(left_indices)
        new_observations, new_actions = augmentData(diff, left_indices,observations,actions)
    else:
        diff = len(left_indices)-len(right_indices)
        new_observations, new_actions = augmentData(diff, right_indices,observations,actions)

    final_observations = np.concatenate((observations,new_observations),axis=0)
    final_actions = np.concatenate((actions,new_actions),axis=0)
    return final_observations, final_actions

################################################################

## Debugging Functions
from inspect import getsource
def code(function):
    print(getsource(function))


