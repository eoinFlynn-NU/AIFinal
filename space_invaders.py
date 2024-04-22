import random
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import GrayScaleObservation
from gymnasium.wrappers import AtariPreprocessing

from matplotlib import pyplot as plt
from collections import deque, namedtuple
import cv2

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
epsilon_d = 0.995
replay_buffer_size = 1000
batch_size = 32
learning_rate = 0.01

# Use render_mode='human' for eval, render_mode='rgb_array' for training
env = gym.make("ALE/SpaceInvaders-v5", render_mode='human', frameskip=1)
env = AtariPreprocessing(env=env, noop_max=30, grayscale_newaxis=True, grayscale_obs=True, screen_size=84)
states = env.observation_space.shape
actions = env.action_space.n

class SpaceInvadersNet(nn.Module):
  def __init__(self, num_frames, num_actions):
    super(SpaceInvadersNet, self).__init__()
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
    x = self.relu(self.conv2(x))
    x = x.view(-1, self.flatten_size)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
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
    
  @staticmethod
  def _process_obs(obs):
    obs = obs.transpose((2, 0, 1))
    return obs

  def select_action(self, state):
    if np.random.rand() < self.epsilon_i:
      return env.action_space.sample()
    else:
      with torch.no_grad():
        q_values = self.model(torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device))
        return q_values.argmax().item()

  def train(self, num_episodes):
    rewards = []
    epsilons = []
    # train for num_episodes
    for episode in range(num_episodes):
      state, info = env.reset()
      self.lives = info['lives']
      state = self._process_obs(np.array(state))
      episode_reward = 0
      terminated = False
      steps = 0
      while not terminated:
        # choose an action with epsilon greedy
        action = self.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        step += 1
        next_state = self._process_obs(np.array(next_state))
        episode_reward += reward

        # add experience to replay memory
        self.memory.append((np.array(state), action, reward, np.array(next_state), terminated))
        state = next_state

        if len(self.memory) >= self.batch_size:
          self.update_model()
        
        if len(self.memory) > 1000:
          self.memory = []
        
        # update target network every {target_update} of frames
        if steps == self.target_update:
          self.target_model.load_state_dict(self.model.state_dict())
          steps = 0
      
      # decary epsilon
      self.epsilon_i = max(self.epsilon_f, self.epsilon_i * self.epsilon_d)
      
      epsilons.append(self.epsilon_i)
      rewards.append(episode_reward)

      

      print(f"Episode: {episode}, Reward: {episode_reward}, Epsilon: {self.epsilon_i}")
    
    # Uncomment to save reward + epsilon values from training to csv files
    # np.savetxt("si_rewards.csv", np.asarray(rewards), delimiter=",")
    # np.savetxt("si_epsilons.csv", np.asarray(epsilons), delimiter=",")

  def update_model(self):
    # choose batch of experiences and convert to tensors
    batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
    states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
    states = np.array(states)
    states = torch.tensor(states, dtype=torch.float32)
    next_states = np.array(next_states)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32, requires_grad=True)
    dones = torch.tensor(dones, dtype=torch.float32)

    # update q values
    with torch.no_grad():
      q_values = self.model(states)
      next_q_values = self.target_model(next_states).max(1)[0].detach()
    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
    selected_q_values = q_values.gather(1, actions.unsqueeze(1))

    # compute loss and back propagate
    loss = F.smooth_l1_loss(selected_q_values, target_q_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
  
  def save_model(self):
    torch.save(self.target_model.state_dict(), 'fw_5H_science.pth')
  
  def load_model(self):
    self.model.load_state_dict(torch.load('si_DQN_CNN_2T_science.pt'))
    model.eval()

model = SpaceInvadersNet(1, actions)
agent = Agent(model, target_update, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_d, batch_size)
# agent.train(2000)
# agent.save_model()

# After training, you can evaluate the performance of the trained agent
def evaluate_agent(num_episodes, weight_file):
    model = SpaceInvadersNet(1, actions)
    model.load_state_dict(torch.load(weight_file))
    model.eval()
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False

        while(not terminated and not truncated):  
          # Ensure state is a valid image
          if state is None:
              print("Invalid state received. Skipping episode.")
              break

          # Resize and normalize the state
          state = cv2.resize(state, (84, 84))
          state = np.array(state, dtype=np.float32) / 255.0

          # Convert state to tensor and add batch dimension
          state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

          # Select best action   
          with torch.no_grad():
            action = model(state_tensor).argmax().item()

          # Execute action
          state,reward,terminated,truncated,_ = env.step(action)
          episode_reward += reward

        
        rewards.append(episode_reward)
        print(f"Episode: {episode}, Reward: {episode_reward}")
    np.savetxt("trained_rewards.csv", np.asarray(rewards), delimiter=",")
    env.close()

weight_file = '' # path to weight file for evaluation 
evaluate_agent(50, weight_file)