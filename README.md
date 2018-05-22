
## Frozen Lake Problem
We were asked to solve this problem using tabular q learning and appromixate q learning. The tabular q learning was a trivial implementation of the q-value bellman backup equations. 

## Acrobat Problem
In this problem, I first did a basic training of policy gradients with a running baseline, using Adam as my optimizer for 2000 episodes. Suprisingly, I got a pretty decent result of 85 +/- 15 steps to pass the line. So, thinking I could do better, I began hyperparameter tuning, starting first with a grid search through L2 weight decay, number of hidden units, and number of trajectories to average before updating the model. I was hoping the weight Decay would slow down my descent so that the model may explore more and find a more optimal solution, rather than jumping directly to the first solution it found. Same thing with number of trajectories. The fewer trajectories I had, the more random I would be, though too much randomness will make it so I never converge to a solution. But, none of the models produced from grid search produced a model with a mean run of below 80, and all the standard deviations were in the 20's.

I then did a grid search over dropout probablities and hidden units. Furthermore, I decided to employ a 3 optimizers. First I would use Adam so I can get out of the 10e4 loss region. Then I would use SGD with momentum and make the learning rate cyclical, so I can bounce out of spiky regions. I would then return to a fresh Adam, so I can avoid the cache that comes with the previous Adam optimizer, since that one must be huge after shrinking the loss from 3e4 to 100. But, this once again, got me around the same results.

I decided then to continuously generate  a neural net with random weights and have it run for 100 episodes. As soon as it experiences a trajectory with a reward of greater than -500, I would return it and use that network for training. Otherwise, I would generate a new network and try again. In this way, I make sure that all the models I train during hyperparameter tuning experience the same number of training episodes, whereas before some would be trained more than others.(Because network doesn't start to train until reward is better than the baseline, which I initially set to -500). I employed this and did a grid search over dropout probablities and hidden units again. But, basically same results.

Finally, I decided to use the network with the smallest average_runs as a starting point for my last exploration. This time, I would bag a model every 100 episodes, in the hopes that I get one which generalizes better than the end results. After a training session is done for one hyperparameter setting, I would evaluate the models for 100 tries and take the model with the smallest average_runs. I would do the same evaluation for all the other "best models" that came out of my grid search. Furthermore, I then had this idea of only updating the network when the reward was greater than the baseline. One of the purposes of the baseline was to change the search from a process of elimination to more of a heuristic search. When the initial baseline is set to -500, the network will only update once the model finds a solution in less than 500 steps. The issue with process of elimination is that while it may decrease the probablity of an undesirable action, updating will cause an increase to the desirable action and the other undesirable action. Whereas with a positive `reward-baseline` you are always increasing the desirable action and decreasing the two other undesirable actions. So I did a grid search across just this "permament baseline". But in the end, all the models I created were basically the same if not worse than the model I achieved on my first try......

* So SGD was prone to blow up, which was why Adam was better

## Lunar Lander
* discovered what it means to write in a functional style
* why I didn't grasp it when using numpy(complexity is about the same with numbers)
* but in practice, top down programming first, functional to refactor
