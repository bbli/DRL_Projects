import getch
# keyboard conttrols
# a -> left, s -> don't move, d -> right
def humanInput():
    invalid = True
    while invalid:
        char = getch.getch()
        # char = char.decode("utf-8") # you need this line if you are running Windows 
                                
        if char == 'a':
            a = 0
            break
        # elif char == 's':
            # a = 1
            # break
        elif char == 'd':
            a = 2
            break
        elif char == 'b':
            invalid = False
    return a


def dataCollector(env):
    observation = env.reset()
    done = False
    step = 0
    actions = []
    observations = [] 

    while not done:
        env.render()
        print(observation)
        
        action = humanInput()
        observation, reward, done, info = env.step(action)

        # store all the observations and actions from one episode
        observations.append(observation)
        actions.append(action)
        
        step += 1
            
    print("Episode finished after {} steps".format(step))
    return observations,actions
