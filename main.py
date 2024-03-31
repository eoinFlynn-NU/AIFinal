import gym
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


# Create the environment
env = gym.make("ALE/Assault-v5")

# Reset the environment

obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

obs = env.reset()
# print("The initial observation is {}".format(obs))
# print(obs[0].shape)
fig, ax = plt.subplots()

def update(i):
    random_action = env.action_space.sample()
    new_obs, reward, terminated, truncated, info = env.step(random_action)
    ax.imshow(env.ale.getScreenRGB())
    if terminated or truncated:
        
        obs, info = env.reset()
        print("Terminated")


ani = FuncAnimation(fig, update, frames=range(100), repeat=False)
plt.show()