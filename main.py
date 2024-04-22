from assault_model import train_assault_agent, evaluate_assault_agent
from freeway import train_freeway_agent, evaluate_freeway_agent
from space_invaders import train_space_invaders_agent, evaluate_space_invaders_agent

import argparse

  
  

def main():
  parser = argparse.ArgumentParser(description='Train or evaluate agents in different atari environments')
  parser.add_argument('env', type=str, help='Name of the atari environment(assault, freeway, or space_invaders)')
  parser.add_argument('mode', type=str, help='train or evaluate')
  parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train or evaluate the agent')
  parser.add_argument('--file_name', type=str, help='File name for the model to be saved to or loaded from')
  args = parser.parse_args()

  if args.env == 'assault':
    if args.mode == 'train':
      train_assault_agent(args.episodes, args.file_name)
    elif args.mode == 'evaluate':
      evaluate_assault_agent(args.file_name, args.episodes)
  if args.env == 'freeway':
    if args.mode == 'train':
      train_freeway_agent(args.episodes, args.file_name)
    elif args.mode == 'evaluate':
      evaluate_freeway_agent(args.episodes, args.file_name)
  if args.env == 'space_invaders':
    if args.mode == 'train':
      train_space_invaders_agent(args.episodes, args.file_name)
    elif args.mode == 'evaluate':
      evaluate_space_invaders_agent(args.episodes, args.file_name)


if __name__ == "__main__":
  main()
