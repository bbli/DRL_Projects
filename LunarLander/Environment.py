import gym
from functions import *
from utils import *
class EnvironmentClass():
    def __init__(self,string):
        # self.env = gym.make(string)
        self.environment = string

    def makeEnvironment(self):
        return gym.make(self.environment)

    def tryEnvironment(self):
        env = gym.make(self.environment)
        state = env.reset()
        count =0
        while True:
            count += 1
            env.render()
            action=humanInput()
            if action == 'b':
                env.close()
                break
            state,reward,done,info = env.step(action)
            if done == True:
                print("Number of steps: ",count)
                print("Reward: ",reward)
                print("Click any key to close the environment")
                getch.getch()
                env.close()
                break

    def showModel(self,model=None):
        # print("This is sampling from the untrained network")
        if model:
            pass
        else:
            model = self.current_model
        env = gym.make(self.environment)
        state = env.reset()
        count =0
        total_reward = 0
        while True:
            count += 1
            env.render()
            action=getAction(model,state)
            state,reward,done,info = env.step(action)
            # time.sleep(0.1)
            total_reward += reward
            if done == True:
                print("Number of steps: ",count)
                print("Reward: ",total_reward)
                print("Click any key to close the environment")
                getch.getch()
                env.close()
                # return count
                break

    def randomWalk(self):
        env = gym.make(self.environment)
        print("This is just random action sampling")
        env.reset()
        total_reward = 0
        while True:
            env.render()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            total_reward += reward
            # time.sleep(0.04)
            if done == True:
                print("Reward: ",total_reward)
                print("Click any key to close the environment")
                getch.getch()
                env.close()
                return total_reward

    # @timeit
    def averageModelRuns(self,model=None,w=None):
        if model:
            pass
        else:
            model = self.current_model
        env = gym.make(self.environment)
        num_trials = 100
        rewards_list = []
        for _ in range(num_trials):
            state = env.reset()
            tr = self.evaluateModel(env,model)
            rewards_list.append(tr)
        rewards_list = np.array(rewards_list)
        mean, std = rewards_list.mean(), rewards_list.std(ddof=1)
        if w:
            w.add_text("Test Scores","Mean Reward: {} Standard Deviation: {}".format(mean,std))

        self.averageModelRuns_rewards_list = rewards_list
        return mean,std
    @staticmethod
    def evaluateModel(env,net):
        state = env.reset()
        net.eval()
        count =0
        total_reward=0
        while True:
            count += 1
            action=getAction(net,state)
            state,reward,done,info = env.step(action)
            total_reward += reward
            if done == True:
                return total_reward



if __name__ == '__main__':
    envir = EnvironmentClass('Acrobot-v1')
