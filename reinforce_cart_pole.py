#Self contained PyTorch implementation of a deep REINFORCE agent
#Suffers from catostrophic interferance / forgetting
from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#Global variables
gamma = 0.999
episodes = 200

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
        pdparam = self.forward(tensor_state) #What are the probability distribution parameters for a given state in tensor form
        pd = Categorical(logits=pdparam) #Action probability distribution
        action = pd.sample() #pi(a|s) in action via pd
        log_prob = pd.log_prob(action) # log_prob of pi(a|s)
        self.log_probs.append(log_prob) # store for training
        return action.item()

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

def sliding_window(data, N):
    """
    :param data: A numpy array, such as return per episode, length M
    :param N: The length of the sliding window
    :return: A numpy array, length M, containing smoothed averaging.
    """
    index = 0
    window = np.zeros(N)
    smoothed = np.zeros(len(data))

    for i in range(len(data)):
        window[index] = data[i]
        index += 1
        smoothed[i] = window[0:index].mean()
        if index == N:
            window[0:-1] = window[1:]
            index = N - 1
    return smoothed

def plot_reward(pi):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('REINFORCE, Cartpole, g=0.999, without a baseline')
    ax1.plot(pi.reward_memory)
    ax1.set_ylabel('Total reward per episode')
    window_size = 100
    means = sliding_window(pi.reward_memory, window_size)
    ax2.plot(means)
    ax2.set_ylabel('Avg. return (Sliding window: 100)')
    plt.show()

def main():
    env = gym.make('CartPole-v0') #Create the OPEN_AI environment
    in_dim = env.observation_space.shape[0] #4
    out_dim = env.action_space.n #2
    pi = Pi(in_dim, out_dim) # Policy pi_theta for REINFORCE - activate policy class
    optimizer = optim.Adam(pi.parameters(), lr=0.01) #lr = learning rate
    for epi in range(episodes):
        state = env.reset()
        for t in range(200): #cartpole max timestep is 200
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if done:
                 break
        loss = train(pi, optimizer) #train per episode
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset() # onpolicy: clear memory after training
        print(f'Episode {epi}.loss:{loss}, \
        total_reward: {total_reward}, solved: {solved}')
        pi.reward_memory.append(total_reward)
    plot_reward(pi)
    env.close()

if __name__ == '__main__':
    main()
