import gym
import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation
from cnn import create_model


# Create the environment
env = gym.make("ALE/Assault-v5")

# Reset the environment
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

# print("The initial observation is {}".format(obs))
# print(obs[0].shape)
#fig, ax = plt.subplots()

height, width, channels = obs_space.shape
actions = action_space.n
obs = env.reset()

# print(type(env.reset()))
# print(f"Height: {height} Width: {width} Channels: {channels} Actions: {actions}")
create_model(obs, height, width, channels, actions, 0.001)

# episodes = 5

# for episode in range(0, episodes):
#     obs = env.reset()
#     terminated = False
#     score = 0
#     num_actions = 0

#     while not terminated:
#     #ax.cla()
#         random_action = env.action_space.sample()
#         new_obs, reward, terminated, truncated, info = env.step(random_action)
#         num_actions += 1
#         score += reward
#     print("Episode: {} Score: {} Actions: {}".format(episode, score, num_actions))
#         #ax.imshow(env.ale.getScreenRGB())
#         # if terminated or truncated:
#         #     obs, info = env.reset()
#         #     print("Terminated")





#ani = FuncAnimation(fig, random_action_update, frames=range(1), repeat=False)
#plt.show()
env.close()