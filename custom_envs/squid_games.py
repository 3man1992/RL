#Test implementation of a navigational agent along a coridoor for mathew daws paper
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

class SquidGames(Env):
    def __init__(self):
        #Actions the agent can take
        self.action_space = Discrete(3) #three actions (go left, stay, go right: 0,1,2)
        #Observation space
        self.observation_space = Box(low = np.array([0]), high = np.array([10]), dtype=np.int32)
        #Starting state
        self.state = 0
        #Length of task
        self.trial_length = 100

    def step(self, action):
        #Apply action
        self.state += action - 1
        #Reduce trial length by 1 second
        self.trial_length -= 1
        #Calculate reward
        if self.state == 10:
            reward =  1
        else:
            reward = 0
        #Check if trial is finished
        if self.trial_length <= 0:
            done = True
        else:
            done = False
        info = {}
        #Return info
        return (self.state, reward, done, info)

    def render(self):
        pass

    def reset(self):
        #Starting state
        self.state = 0
        #Length of task
        self.trial_length = 100
env = SquidGames()
print(env.observation_space.sample())

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        #env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))