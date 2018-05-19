import gym
from functions import *
from utils import *
class EnvironmentClass():
    def __init__(self,string):
        # self.env = gym.make(string)
        self.environment = string

    def tryEnvironment(self):
        env = gym.make(self.environment)
        state = env.reset()
        count =0
        while True:
            count += 1
            env.render()
            action=humanInput()
            if action == 3:
                env.close()
                break
            state,reward,done,info = env.step(action)
            if done == True:
                print("Number of steps: ",count)
                print("Click any key to close the environment")
                getch.getch()
                env.close()
                break

    def showModel(self,net):
        # print("This is sampling from the untrained network")
        env = gym.make(self.environment)
        state = env.reset()
        count =0
        while True:
            count += 1
            env.render()
            action=getAction(net,state)
            state,reward,done,info = env.step(action)
            # time.sleep(0.1)
            if done == True:
                print("Click any key to close the environment")
                getch.getch()
                env.close()
                # return count
                print("Number of steps: ",count)
                break

    def randomWalk(self):
        env = gym.make(self.environment)
        print("This is just random action sampling")
        env.reset()
        while True:
            env.render()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            time.sleep(0.04)
            if done == True:
                print("Click any key to close the environment")
                getch.getch()
                env.close()
                return 0

    # @timeit
    def averageModelRuns(self,model,w=None):
        env = gym.make(self.environment)
        num_trials = 100
        counts_list=[]
        for _ in range(num_trials):
            state = env.reset()
            count = self.evaluateModel(env,model)
            counts_list.append(count)
        counts_list = np.array(counts_list)
        mean, std = counts_list.mean(), counts_list.std(ddof=1)
        if w:
            w.add_text("Test Scores","Mean: {} Standard Deviation: {}".format(mean,std))
        return mean,std
    @staticmethod
    def evaluateModel(env,net):
        state = env.reset()
        net.eval()
        count =0
        while True:
            count += 1
            action=getAction(net,state)
            state,reward,done,info = env.step(action)
            if done == True:
                return count



if __name__ == '__main__':
    envir = EnvironmentClass('Acrobot-v1')
