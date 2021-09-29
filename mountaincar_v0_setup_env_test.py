import gym
env = gym.make('MountainCar-v0')

#Enviromental investigative functions for different envs
print(env.action_space) #How many actions are there and is it discrete or continous
print(env.observation_space) #What is the observation dimensionality in a box space
print(env.observation_space.high) #
print(env.observation_space.low)

# for i_episode in range(20):
#     observation = env.reset() #Produces an initial observation
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
