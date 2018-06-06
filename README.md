# README
---
## Intro
This is all the code that I wrote for the deep reinforcement learning class I took in my last quarter at UCSB.
## Running the Models
Acrobot:

`python Acrobot/visualize.py`

LunarLander:

`python LunarLander/visualize.py`

For the MountainCar, Pendulum, and FrozenLake environments, I did not save trained models. The former two because I couldn't get a respectable result after training, and the latter because it was fairly fast to train and not much hyperparameter tuning was nesscary. The following are the corresponding scripts to run my current machine learning algorithm, for all environments.

MountainCar:

`python MountainCar/imitation_learning_starter.py`

FrozenLake:

`python FrozenLake/aql_starter.py`

Acrobot:

`python Acrobot/pg_starter.py`

LunarLander:

`python LunarLander/baseline_main.py`

`python LunarLander/actor_main.py`

Pendulum:

`python Pendulum/main.py`


## What I learned
### MountainCar
This was my first time training a regular neural net. So to no suprise, I made a rookie mistake in intializing a overly complex model for this problem. I know this because my loss function would barely improve when I had 4 hidden layers, and because for all the subsequent environments, I achieved sucess with just one or two hidden layers. 

Because this was the first assigment, I was rather determined to get a working model and got extremely irritated when nothing that I tried worked. Eventually, I got fed up with my guess and check process and instead took time out to learn how to use tensorboard. While I still couldn't sucessfully train a model in the end(I suppose this wasn't a suprise given I only trained for 3 episodes), the tensorboard was a plus, as I end up using it heavily for hyperparameter tuning in the following environments.

### Frozen Lake
We were asked to solve this problem using tabular q learning and appromixate q learning. The tabular q learning code was an easy implementation of the q-value bellman backup equations. As for the appromixate q learning, implementation was also fairly straightforward, but unlike the q table code, I was getting 0 rewards when I first ran the code and my weights were blowing up....

* linear network -> onehot encoding as perfect nonlinear transformation, equivalence with tabular q learning with no momentum, no hidden layers, 0 weight initalization. 

### Acrobat Problem
This may be a bit long, as for this environment I actually made a concious effort to track what I did.

In this problem, I first did a basic training of policy gradients with a running baseline, using Adam as my optimizer for 2000 episodes. Suprisingly, I got a pretty decent result of 85 +/- 15 steps to pass the line. So, thinking I could do better, I began hyperparameter tuning, starting first with a grid search through L2 weight decay, number of hidden units, and number of trajectories to average before updating the model. I was hoping the weight Decay would slow down my descent so that the model may explore more and find a more optimal solution, rather than jumping directly to the first solution it found. Same thing with number of trajectories. The fewer trajectories I had, the more random I would be, though too much randomness will make it so I never converge to a solution. But, none of the models produced from grid search produced a model with a mean run of below 80, and all the standard deviations were in the 20's.

I then did a grid search over dropout probablities and hidden units. Furthermore, I decided to **use 3 optimizers**, as a loose way to reset the learning rate. First I would use Adam so I can get out of the 10e4 loss region. Then I would use SGD with momentum and make the learning rate cyclical, so I can bounce out of spiky regions. I would then return to a fresh Adam, so I can avoid the cache that comes with the previous Adam optimizer, since that one must be huge after shrinking the loss from 3e4 to 100. But, this once again, got me around the same results despite what I thought was a strong theoretical basis.

I decided then to continuously generate  a neural net with random weights and have it run for 100 episodes. As soon as it experiences a trajectory with a reward of greater than -500, I would return it and use that network for training. Otherwise, I would generate a new network and try again. In this way, I make sure that all the models I train during hyperparameter tuning **experience the same** number of training episodes, whereas before some would be trained more than others.(Because network doesn't start to train until reward is better than the baseline, which I initially set to -500). I employed this and did a grid search over dropout probablities and hidden units again. But, basically same results.

Finally, I decided to use the network with the smallest average_runs as a starting point for my last exploration. This time, I would **bag a model** every 100 episodes, in the hopes that I get one which generalizes better than the end results. After a training session is done for one hyperparameter setting, I would evaluate the models for 100 tries and take the model with the smallest average_runs. I would do the same evaluation for all the other "best models" that came out of my grid search. Furthermore, I then had this idea of only updating the network when the **reward was greater than the baseline.** One of the purposes of the baseline was to change the search from a process of elimination to more of a heuristic search. The issue with process of elimination is that while it may decrease the probablity of an undesirable action, it will cause an increase to the desirable action and the other undesirable action. Whereas with a positive `reward-baseline` you are always increasing the desirable action and decreasing the two other undesirable actions. I combined this with what I call a **mixed policy**: if the highest probability the neural net spits out is above a certain threshold, I will switch from a stochastic policy to a deterministic one. With these two made up tricks, I managed to lower the standard deviation of the step count as follows:


| | BaselineTune | NO BaselineTune|
|---| --- | ----|
|Mixed | 12 | 25|
|Stochastic | 24 | 40|

(Standard deviation obtained from running each combination 100 times)

### Lunar Lander
**Baseline**
* realized that shallow networks are better, at least for policy gradients, because deeper networks tend to get stuck in local mins.
    * single layer works better, even though I have been told single layer just remembers-> no issue for just one environment right? -> Same as Adam>SGD
* solved the environment(209+50), even though I didn't expect it to because it was too "simple" of a model...my mixed policy trick help immensely here, as without it, final model will only reach a mean of 85.

* Experiment/Environment classes
    * Experiment for hyperparameter tuning and logging
    * Environment for evaluation and display

**Actor Critic**
* I think that until I can get the advantage to be positive, I will never get a good solution -> actually had to increase learning rate, which I was wary about because it may cause overfitting.
* high learning rate cause probablity to go to one
* dividing reward by 10 so critic doesn't chase the outlier

Although this wasn't assigned for class, I decided to use this environment to test out actor-critic instead of Acrobot, since I already got rather a rather optimal result from just policy gradients with baseline(the model moved the joint to the desired height in less than a second). But before implementating the batch actor-critic algorithm, I decided to train a baseline model on this environment as a control case. Furthermore, I finished implementating my abstract Environment and Experiment classes, which I had started in the Acrobot project. The purpose of the Environment class is to provide functionality for evaluating and displaying models, and the Experiment class is where the training algorithm lives. Before, the training algorithm was inside a function, which I called for hyperparameter tuning. The issue with this is that I have lost most of the state informationinside, as the function only returns the trained model. And seeing as state is crucial for debugging, I put this function inside the method of a class so that I can "log" desired variables as attributes of the Experiment object.

Because I thought this environment was more complex, I started the neural net out at 2 hidden layers rather than one. But, much to my dismay, these neural nets performed horribly, barely acheiving a positive evaluation reward. Once I changed back to one hidden layer, the reward skyrocketed to 80. I found this suprising, because I have been told that more hidden layers and less neurons per layer were better than vice versa, as the latter tends to "remember" datapoints and so has poor generalization. But I suppose I don't need to generalize in this scenario, as I am always training in the same environment. Also, another training tip from supervised learning that doesn't seem to be true in reinforcement learning is SGD with momentum being better then Adam, since Adam is prone to overfitting. But in practice, Adam has always worked out better for me in this environments, probably for the same reason that shallow nets do; because I don't need to generalize. And using my **mixed policy trick from above on top of a shallow net**, I even managed to solve the environment, getting a mean evaluation reward of 209!

### Pendulum

* pretty much followed paper, with target networks, memory buffer, etc.
* tensorboard's inability to search well
* tbh, how do people get actor critic to even work -> point on my concerns, on top of the overfitting problem above.
