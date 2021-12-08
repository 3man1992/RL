"""Note that is an MDP implementation such that state == observation, this is not a POMDP"""

#Test implementation of a squid games
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import style
import cv2
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim

#Constants
grid_shape = 10
show_every = 10
gamma = 0.999
episodes = 100
trial_rewards = []
learning_rate = 0.01
trial_length = 100

#Define the environment
class SquidGames(Env):
    def __init__(self):
        #Actions the agent can take
        self.action_space = Discrete(5) #5 actions (go left, stay, go right, up, down: 0,1,2,3,4)
        #Observation space - not really used though eh?
        self.observation_space = Box(low = 0, high = 1, shape = (grid_shape, grid_shape), dtype = np.uint8)
        #Set up grid and starting state
        self.state = np.zeros((grid_shape, grid_shape))
        #Set trial length
        self.trial_length = trial_length

    def step(self, action):
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
        if np.any(self.state[0][:]):
            reward = 1
        elif np.any(self.state[-1][:]):
            reward = random.randint(0, 2)
        else:
            reward = 0

        #Conditions for finishing a trial
        if self.trial_length <= 0:
            done = True
        else:
            done = False

        #Return info
        info = {}

        #print state grid_shape
        # print('State shape action', self.state.shape)
        return (self.state, reward, done, info)

    def render(self):
        #Recapitulate statespace into an RGB array for rending
        render_coords = np.zeros((grid_shape,grid_shape, 3), dtype = np.uint8)
        animal_coords = np.nonzero(self.state)
        x = animal_coords[0][0]
        y = animal_coords[1][0]
        render_coords[x][y] = (255, 150, 0)
        img = Image.fromarray(render_coords, "RGB")
        img = img.resize((500, 500))
        cv2.imshow("Custom RL env", np.array(img))
        #waits for user to press any key (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(1) #ms to wait
        pass

    def reset(self):
        #Wipe state
        self.state = np.zeros((grid_shape, grid_shape), dtype = np.uint8)
        #Starting state
        self.state[random.randint(0, grid_shape-1)][random.randint(0, grid_shape-1)] = 1
        #Length of task
        self.trial_length = 100
        #print state grid_shape
        # print('State shape reset', self.state.shape)
        return(self.state)

#############################################################################################
# """Test simple run"""
# env = SquidGames()
# for epi in range(episodes):
#     state = env.reset()
#     for t in range(trial_length):
#         env.render()
#         action = env.action_space.sample()
#         next_state, reward, done, info = env.step(action)
#         print(reward)
#         if done:
#             break

#############################################################################################
# #Define the Agent
class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__() #I still don't understand why super is used here in the parent class
        layers = [
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        ] #Alter this variable to modify the architecture of the neural network
        self.model = nn.Sequential(*layers) #Creates the model with an architecture as defined by the order of the variable layers
        self.onpolicy_reset() #The log probabilites and rewards attached to the policy are reset and set to empty
        self.train() #set the module in training mode, mode=True as default
        self.reward_memory = []

    #A method of reseting the policy by setting log probs and rewards to empty
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    #Define how to get the output of the neural network
    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    #Define the method to produce actions
    def act(self, state):
        tensor_state = torch.from_numpy(state.astype(np.float32)) # Creates a Tensor from a numpy.ndarray
        tensor_state = torch.flatten(tensor_state)
        # print('Flatten tensor state', tensor_state)
        pdparam = self.forward(tensor_state) #What are the probability distribution parameters for a given state in tensor form
        # print('pdparam', pdparam)
        pd = Categorical(logits = pdparam) #Action probability distribution
        # print('test', pd.probs) #What is the probability of choosing each action given this state
        action = pd.sample() #pi(a|s) in action via pd
        # print('Test action', action)
        log_prob = pd.log_prob(action) # log_prob of pi(a|s)
        self.log_probs.append(log_prob) # store for training
        return action #Returns an action

# #Train archy the blob mouse
def train(pi, optimizer):
    #Inner gradient-ascent loop of REINFORCE algorithm
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32) #the returns
    future_ret = 0.0
    # compute the returns efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    # baseline = rets - rets.mean() - baseline seems not to work
    # rets = rets - baseline # so commenting out code
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs) #Concatenates a sequence of tensors
    loss = - log_probs * rets #gradient term, negative for maximising
    loss = torch.sum(loss)
    optimizer.zero_grad() #Sets gradients of all model parameters to zero
    loss.backward() # backpropagate, compute gradients - Policy gradient
    optimizer.step() #gradient-ascent, update the weights / update the policy parameters
    return loss

def main():
    env = SquidGames() #Create the custom open_AI env
    # in_dim = env.state.shape # what dimensionality is it? potential bug
    # in_dim = ([10, 10])
    # in_dim = env.observation_space.shape
    in_dim = 100 # (Shape (10,10) flattened as pytorch errors with 2d input)
    # print('input_dimension:', in_dim)
    out_dim = env.action_space.n
    # print('output_dimension:', out_dim)
    pi = Pi(in_dim, out_dim) # Policy pi_theta for REINFORCE - activate policy class
    optimizer = optim.Adam(pi.parameters(), lr=learning_rate) #lr = learning rate
    for epi in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            env.render()
            action = pi.act(state)
            n_state, reward, done, info = env.step(action)
            pi.rewards.append(reward)
            episode_reward += reward #For plotting purposes
            if done:
                break

        #Tally up rewards after trial
        total_reward = sum(pi.rewards)
        pi.reward_memory.append(total_reward)

        #Update policy
        loss = train(pi, optimizer) #train per episode
        pi.onpolicy_reset() # onpolicy: clear memory after training

        #Log process
        solved = total_reward > 65 #65% of time is spent in reward zone
        trial_rewards.append(episode_reward)
        print(f'Episode {epi}, loss:{loss}, total_reward: {total_reward}, solved: {solved}')
    env.close()

if __name__ == '__main__':
    main()
    moving_avg = np.convolve(trial_rewards, np.ones((show_every)) / show_every, mode="valid")
    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f"reward {show_every}")
    plt.xlabel("episode #")
    plt.show()
