from assault_overheat_update import train_assault_agent, evaluate_assault_agent
import argparse

  
  

def main():
  parser = argparse.ArgumentParser(description='Train or evaluate agents in different atari environments')
  parser.add_argument('env', type=str, help='Name of the atari environment(assault, freeway, or space_invaders)')
  parser.add_argument('mode', type=str, help='train or evaluate')
  parser.add_argument('--train_episodes', type=int, default=1000, help='Number of episodes to train the agent')
  parser.add_argument('--save_file_name', type=str, default='assault_model.pth', help='File name to save the trained model')
  parser.add_argument('--eval_episodes', type=int, default=50, help='Number of episodes to evaluate the agent')
  parser.add_argument('--load_file_name', type=str, default='assault_model.pth', help='File name to load the trained model')
  args = parser.parse_args()

  if args.env == 'assault':
    if args.mode == 'train':
      train_assault_agent(args.num_episodes, args.save_file_name)
    elif args.mode == 'evaluate':
      evaluate_assault_agent(args.load_file_name, args.num_eval_episodes)


if __name__ == "__main__":
  main()
