import random
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import GrayScaleObservation

from matplotlib import pyplot as plt
from collections import deque, namedtuple
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
from copy import deepcopy

from torchviz import make_dot

# Hyperparameters
num_episodes = 1000
max_steps_per_episode = 1000
target_update = 2000
gamma = 0.99
epsilon_i = 1.0
epsilon_f = 0.01
epsilon_d = 0.99
replay_buffer_size = 10000
batch_size = 32
learning_rate = 0.01
# frame_stack = 4
# Use num_lives to reset after each life (maybe)
num_lives = 4

episodes_list = []
rewards_list = []
epsilon_list = []


env = gym.make("ALE/Assault-v5", render_mode='rgb_array')
env = GrayScaleObservation(env, keep_dim=True)
states = env.observation_space.shape
actions = env.action_space.n
# observation_space = Box(0, 255, (4, 210, 160), uint8)

class AssaultNet(nn.Module):

  def __init__(self, num_frames, num_actions):
    super(AssaultNet, self).__init__()
    self.num_frames = num_frames
    self.num_actions = num_actions

    self.conv1 = nn.Conv2d(
      in_channels=1,
      out_channels=32,
      kernel_size=8,
      stride=4,
    )
    self.conv2 = nn.Conv2d(
      in_channels=32,
      out_channels=64,
      kernel_size=4,
      stride=2
    )
    self.flatten_size = self._get_flatten_size(num_frames)
    self.fc1 = nn.Linear(
      in_features=self.flatten_size,
      out_features=512
    )
    self.fc2 = nn.Linear(
      in_features=512,
      out_features=256
    )
    self.fc3 = nn.Linear(
      in_features=256,
      out_features=num_actions
    )
    self.relu = nn.ReLU()

  def _get_flatten_size(self, input_channels):
    with torch.no_grad():
      fake_input = torch.zeros((1, 1, 84, 84))
      fake_output = self.conv2(self.conv1(fake_input))
      return fake_output.view(-1).size(0)
  
  def forward(self, x):

    x = self.relu(self.conv1(x))
    # print(np.array(x).shape)
    x = self.relu(self.conv2(x))
    # print(np.array(x).shape)
    x = x.view(-1, self.flatten_size)
    # print(np.array(x).shape)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

class Agent:
  def __init__(self, model, target_update, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_d, batch_size):
    self.model = model
    self.target_model = deepcopy(model)
    self.target_update = target_update

    self.gamma = gamma
    self.epsilon_i = epsilon_i
    self.epsilon_f = epsilon_f
    self.epsilon_d = epsilon_d

    self.loss = nn.SmoothL1Loss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    self.batch_size = batch_size
    self.memory = []

    self.cuda = True if torch.cuda.is_available() else False
    self.device = torch.device("cuda" if self.cuda else "cpu")
    model.to(self.device)
    
  @staticmethod
  def _process_obs(obs):
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    obs = obs[:, :, np.newaxis]
    obs = obs.transpose((2, 0, 1))
    return obs

  def _crop_frame(self, obs):
    obs = obs[40:-20, :, :]
    return obs

  def select_action(self, state):
    if np.random.rand() < self.epsilon_i:
      return env.action_space.sample()
    else:
      with torch.no_grad():
        q_values = self.model(torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device))
        return q_values.argmax().item()

  def train(self, num_episodes):
    frame_count = 0
    for episode in range(num_episodes):
      state, info = env.reset()
      unchanged_state = state
      #state = self._crop_frame(state)
      state = self._process_obs(np.array(state))
      episode_reward = 0
      num_lives = 4
      overheat = False
      terminated = False
      while not terminated:
        frame_count += 1
        if unchanged_state[197, 150, :].all() == 0:
          action = self.select_action(state)
        else:
          if action not in [0, 3, 4]:
            overheat = True
            action = np.random.choice([0, 3, 4])

        next_state, reward, terminated, truncated, info = env.step(action)
        if overheat:
          reward = -5
          overheat = False
        unchanged_state = next_state

        #next_state = self._crop_frame(next_state)
        next_state = self._process_obs(np.array(next_state))

        if info['lives'] < num_lives:
          reward = -50
          num_lives -= 1
        # print(episode_reward)
        # print(info['lives'])
        episode_reward += reward
        self.memory.append((np.array(state), action, reward, np.array(next_state), terminated))
        state = next_state

        if len(self.memory) >= self.batch_size:
          self.update_model()
        
        if len(self.memory) > 1000:
          self.memory = self.memory[32:]
          pass
      
      self.epsilon_i = max(self.epsilon_f, self.epsilon_i * self.epsilon_d)

      if frame_count % self.target_update == 0:
        self.target_model.load_state_dict(self.model.state_dict())
        frame_count = 0

      print(f"Episode: {episode}, Reward: {episode_reward}, Epsilon: {self.epsilon_i}")

      epsilon_list.append(self.epsilon_i)
      episodes_list.append(episode)
      rewards_list.append(episode_reward)
    #torch.save(self.model.state_dict(), "assault_DQN_CNN_1T.pt")

  def update_model(self):
    batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
    #batch = range(len(self.memory))
    states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
    states = np.array(states)
    states = torch.tensor(states, dtype=torch.float32)
    next_states = np.array(next_states)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32, requires_grad=True)
    dones = torch.tensor(dones, dtype=torch.float32)

    with torch.no_grad():
      q_values = self.model(states)
      next_q_values = self.target_model(next_states).max(1)[0].detach()
    
    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
    selected_q_values = q_values.gather(1, actions.unsqueeze(1))

    # print("q_values.requires_grad:", q_values.requires_grad)
    # print("actions.requires_grad:", actions.requires_grad)
    # print("selected_q_values.requires_grad:", selected_q_values.requires_grad)

    loss = F.smooth_l1_loss(selected_q_values, target_q_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
  
  def save_model(self, name):
    torch.save(self.target_model.state_dict(), name)
  
  def load_model(self, name):
    self.model.load_state_dict(torch.load(name))
    self.model.eval()

model = AssaultNet(1, actions)
agent = Agent(model, target_update, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_d, batch_size)
agent.train(500)
agent.save_model('assault_model_with_target_fix500.pth')


with open('assault_data_target_model_fixed.csv', mode='w') as asteroid_file:
  asteroid_writer = csv.writer(asteroid_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  asteroid_writer.writerow(['Episode', 'Reward', 'Epsilon'])
  for i in range(len(episodes_list)):
    asteroid_writer.writerow([episodes_list[i], rewards_list[i], epsilon_list[i]])

# After training, you can evaluate the performance of the trained agent
def evaluate_agent(agent, num_episodes=50):
    total_rewards = []
    global epsilon_i
    epsilon_i = 0
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        while not done:
            if state[197, 150, :].all() == 0:
              action = agent.select_action(state)
            else:
              if action not in [0, 3, 4]:
                action = np.random.choice([0, 3, 4])
            #action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    average_reward = sum(total_rewards) / num_episodes
    print(f"Average reward over {num_episodes} episodes: {average_reward}")


# model = AssaultNet(1, actions)
# agent = Agent(model, target_update, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_d, batch_size)
# agent.load_model("assault_trained_model_with_overheat1000.pth")
# evaluate_agent(agent)




# Evaluate the trained agent
# evaluate_agent(agent)
