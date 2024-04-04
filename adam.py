import random
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import GrayScaleObservation

from matplotlib import pyplot as plt
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
num_episodes = 1000
max_steps_per_episode = 1000
target_update = 1000
gamma = 0.99
epsilon_i = 1.0
epsilon_f = 0.01
epsilon_d = 0.99
replay_buffer_size = 10000
batch_size = 32
learning_rate = 0.01
frame_stack = 4



env = gym.make("ALE/Assault-v5", render_mode='human')
env = GrayScaleObservation(env)
env = FrameStack(env, frame_stack)
states = env.observation_space.shape[0]
actions = env.action_space.n
# observation_space = Box(0, 255, (4, 210, 160), uint8)

class AssaultNet(nn.Module):

  def __init__(self, num_frames, num_actions):
    super(AssaultNet, self).__init__()
    self.num_frames = num_frames
    self.num_actions = num_actions

    self.conv1 = nn.Conv2d(
      in_channels=num_frames,
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
      out_features=num_actions
    )
    self.relu = nn.ReLU()

  def _get_flatten_size(self, input_channels):
    with torch.no_grad():
      fake_input = torch.zeros((1, input_channels, 210, 160))
      fake_output = self.conv2(self.conv1(fake_input))
      return fake_output.view(-1).size(0)
  
  def forward(self, x):

    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = x.view(-1, self.flatten_size)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x

class Agent:
  def __init__(self, model, target_update, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_d, batch_size):
    self.model = model
    self.target_model = model
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
    
    
  def select_action(self, state):
    if np.random.rand() < self.epsilon_i:
      return env.action_space.sample()
    else:
      with torch.no_grad():
        q_values = self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return q_values.argmax().item()

  def train(self, num_episodes):
    for episode in range(num_episodes):
      print(episode)
      state = env.reset()
      episode_reward = 0
      terminated = False
      while not terminated:
        action = self.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        print(episode_reward)
        print(info['lives'])
        episode_reward += reward
        if np.array(state).shape == (4, 210, 160):
          self.memory.append((np.array(state), action, reward, np.array(next_state), terminated))
        state = next_state

        if len(self.memory) >= self.batch_size:
          self.update_model()
      
      self.epsilon_i = max(self.epsilon_f, self.epsilon_i * self.epsilon_d)

      if episode % self.target_update == 0:
        self.target_model.load_state_dict(self.model.state_dict())

      print(f"Episode: {episode}, Reward: {episode_reward}")

  def update_model(self):
    batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
    states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
    states = np.array(states)
    states = torch.tensor(states, dtype=torch.float32)
    # print("Shape of states after tensor conversion:", states.shape)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = np.array(next_states)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = self.model(states)
    next_q_values = self.target_model(next_states).max(1)[0].detach()
    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

    selected_q_values = q_values.gather(1, actions.unsqueeze(1))

    loss = F.smooth_l1_loss(selected_q_values, target_q_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

model = AssaultNet(frame_stack, actions)
agent = Agent(model, target_update, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_d, batch_size)
agent.train(10)


# After training, you can evaluate the performance of the trained agent
def evaluate_agent(agent, num_episodes=3):
    total_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    average_reward = sum(total_rewards) / num_episodes
    print(f"Average reward over {num_episodes} episodes: {average_reward}")

# Evaluate the trained agent
evaluate_agent(agent)
