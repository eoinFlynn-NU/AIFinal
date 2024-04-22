from assault_model import train_assault_agent, evaluate_assault_agent
import argparse

  
  

def main():
  parser = argparse.ArgumentParser(description='Train or evaluate agents in different atari environments')
  parser.add_argument('env', type=str, help='Name of the atari environment(assault, freeway, or space_invaders)')
  parser.add_argument('mode', type=str, help='train or evaluate')
  parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train or evaluate the agent')
  parser.add_argument('--save_file_name', type=str, help='File name to save the trained model')
  parser.add_argument('--load_file_name', type=str, help='File name to load the trained model')
  args = parser.parse_args()

  if args.env == 'assault':
    if args.mode == 'train':
      train_assault_agent(args.episodes, args.save_file_name)
    elif args.mode == 'evaluate':
      evaluate_assault_agent(args.load_file_name, args.episodes)
  # if args.env == 'freeway':
  #   if args.mode == 'train':
  #     train_freeway_agent(args.episodes, args.save_file_name)
  #   elif args.mode == 'evaluate':
  #     evaluate_freeway_agent(args.load_file_name, args.episodes)
  # if args.env == 'space_invaders':
  #   if args.mode == 'train':
  #     train_space_invaders_agent(args.episodes, args.save_file_name)
  #   elif args.mode == 'evaluate':
  #     evaluate_space_invaders_agent(args.load_file_name, args.episodes)


if __name__ == "__main__":
  main()
