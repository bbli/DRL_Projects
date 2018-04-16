## Questions
> Should I use L2 regularization?
    * No, I want to overfit as much as possible
> How complex is this task/increase number of hidden units
> is there not a max step?
    * It probably is in the gym code

* Probably should use SGD because I don't have much training points
* Or use Adam with a high learning rate

* to get out of initial steps, don't click "a" on first pass

## TODO
> check what comes out of network/ what type the label is
    * yeah it seems I am being too careless about this

> write eval code

> test difference between argmax methods
    * its the same

> is my loss number supposed to be this high?
    * yes, theoretically is a **lower** bound, not upper

> switch to cross entropy loss, so output of network is softmax

> Make decisions probabilistic instead of argMax
> define accuracy function

> change weight initalization/ rescale datapoints??
    * rescale first as that may have bigger effect

> I think best thing to do is to define a scheduler, because I really need a high learning rate in begining


* class imbalence problem
    * hmm, for some reason, network is not updating that much after I implemented this
    * in particular, the accuracy is always 0.5 now
* remember rule of thumb: parameters = datapoints/10


## Future
> learn about value iteration on MDP2

> how does weight decay promote smoothness
* get ipython to work with gym
* use tensorboard to see loss and weight update ratio
    * also text regarding network parameters

* test whether or not smaller networks mean larger weight updates
* learn how to define my own loss function(see ML Improvers)
    * this guy defines a dice coefficient loss function subclassing from `nn.Module` not `torch.autograd.Function`. What is the difference?
