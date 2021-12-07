#Test implementation of a squid games
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import style
import cv2

#Constants
grid_shape = 10

class SquidGames(Env):
    def __init__(self):
        #Actions the agent can take
        self.action_space = Discrete(5) #5 actions (go left, stay, go right, up, down: 0,1,2,3,4)
        #Observation space - not really used though eh?
        self.observation_space = Box(low = 0, high = 0, shape = (grid_shape,grid_shape), dtype=np.int32)
        #Set up grid and starting state
        self.state = np.zeros((grid_shape,grid_shape,3), dtype = np.uint8)
        self.state[random.randint(0,grid_shape-1)][random.randint(0,grid_shape-1)] = (255, 175, 0)
        #Length of task
        self.trial_length = 100

    def step(self, action):
        """Actions"""
        #Don't move
        if action == 0:
            pass
        #Move down
        if action == 1:
            self.state = np.roll(self.state, [1,0], axis = (0,1))
        #Move up
        if action == 2:
            self.state = np.roll(self.state, [-1,0], axis = (0,1))
        #Move left
        if action == 3:
            self.state = np.roll(self.state, [0,-1], axis = (0,1))
        #Move right
        if action == 4:
            self.state = np.roll(self.state, [0, 1], axis = (0,1))

        #Reduce trial length by 1 second
        self.trial_length -= 1

        #Calculate reward - need to implement
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
        img = Image.fromarray(self.state, "RGB")
        img = img.resize((500, 500))
        cv2.imshow("Custom RL env", np.array(img))
        #waits for user to press any key
        #(this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)
        #closing all open windows
        cv2.destroyAllWindows()

    def reset(self):
        #Starting state
        self.state[random.randint(0,grid_shape-1)][random.randint(0,grid_shape-1)] = (255, 175, 0)
        #Length of task
        self.trial_length = 100
env = SquidGames()
# print(env.state)
# print(env.observation_space.sample())

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
