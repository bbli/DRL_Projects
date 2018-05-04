#=================================================================================  
# Simple example of using gradient descent to minimize a loss function 
# 
# The code was initially written for UCSB Deep Reinforcement Learning Seminar 2018
#
# Authors: Jieliang (Rodger) Luo, Sam Green
#
# April 20th, 2018
#=================================================================================

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

# Build a one-layer linear network
# 16 inputs, 4 outputs
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc1(x)
        return x

# Instantiate the Net class
net = Net()

# Set the first entry to 1. and the rest to 0.
# [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
x = np.identity(16)[0]
print(x)

# Convert x to PyTorch Variable
x = Variable(torch.FloatTensor(x))

# Let's see what our (untrained) network thinks x is.
# y_hat will have four entries.
y_hat = net(x)
print(y_hat)

# Now lets train the network to associate x with some arbitrary specific value (934) 
#   at the 3rd entry of y
y = Variable(torch.FloatTensor([934]))

# This instantiates a PyTorch mean squared error loss function object.
# http://pytorch.org/docs/master/nn.html#torch.nn.MSELoss
loss_function = nn.MSELoss()

learning_rate = .01

# Apply gradient descent to make net(x)[2]=934
for epoch in range(1000):
	y_hat = net(x)
	print(y_hat)

	# Calculate the current loss (want y_hat[2] to be equal to 934). You can think
	#   about this graphically (see last slide from 4/19) where only the y_hat[2]
	#  	node is connected to the loss function. Therefore, the gradient will only
	#   flow this node, and not through the other nodes of y_hat.
	loss = loss_function(y_hat[2], y)
	print(loss)

	# Standard bookkeeping. Zero the gradients before running the backward pass
	net.zero_grad()

	# Calculate the gradient of the loss with respect to the NN parameters.
	# The partial derivatives are stored in each parameter (accessed below)
	loss.backward()

	# Update the weights using gradient descent
	for param in net.parameters():
		param.data -= learning_rate * param.grad.data