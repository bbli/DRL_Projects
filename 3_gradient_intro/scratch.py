import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *

from functions import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6,20)
        # self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(20,3)
    def forward(self, x):
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        # print(x)
        # print("Shape: {}".format(x.shape))
        return F.softmax(x)

net = Net()
net.load_state_dict(torch.load('May04-19:11.pt'))

# count = evaluateModel(net)
# print(count)
print(averageModelRuns(net))

data = torch.ones(5,6)
data = Variable(data).float()
################ **Learning about Schedulers** ##################
# learn_rate = 10
# optimizer = optim.SGD(net.parameters(),lr=learn_rate)
# # ## not a copy!!
# # group = next(iter(optimizer.param_groups))
# # group['initial_lr'] = learn_rate

# ## For this problem, I want to step the scheduler along with the optimizer
# ## so every **episode**
# ## probably will set period to around 80
# def cyclic(period):
    # def f(episode):
        # modulus = episode % period
        # return 1/(1+0.05*modulus)
    # return f

# scheduler = LambdaLR(optimizer,lr_lambda=cyclic(80))
# group = next(iter(optimizer.param_groups))
# for epoch in range(100):
    # print(optimizer.state_dict()['param_groups'][0]['lr'])
    # # print(group['lr'])
    # scheduler.step()

################ **Saving results into a DataFrame** ##################
# probability_parameters = [0.3,0.4,0.5]
# neuron_parameters = [20,30,40,50]
# x,y = len(probability_parameters), len(neuron_parameters)
# average_run_table = np.zeros((x,y))
# std_table = np.zeros((x,y))
# for i,prob in enumerate(probability_parameters):
    # for j,neuron in enumerate(neuron_parameters):
        # average_runs= i+j
        # average_run_table[i,j] = average_runs

