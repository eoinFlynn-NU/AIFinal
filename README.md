# Deep Q-Learning in Atari
Our project was focused on utilizing deep q-learning algorithms to train an AI agent to be able to play a variety of retro Atari games. The goal was to design a model that could be trained using the existing gymnasium environments to outperform an agent picking entirely random moves, and ideally outperform a human. While improving the model’s ability to play Atari games is not the most pressing problem in the world, it acts as a good way to see how reinforcement learning can be used to optimize AI agents in a more general context. Also, by deciding to focus on a variety of different games, we are hoping to gain more insights as to how the model is actually learning and what kind of strategies it develops under different game environments.

We have set up the games assault, freeway, and space invaders as demos. For each game, you can pick a number of episodes and then train and evaluate each model.

## Codebase + Environment Setup
To run first pip install the following packages
- opencv-python
- pytorch
- gymnasium
- assault, freeway, and space invaders gym enviroments

## Running Demo
You can run a demo of the project with the command ```python main.py``` followed by the 4 arguments environment, mode, episodes, and file_name. We have provided pretrained weights for each game that can be loaded into the model.
- environment specifies which atari game will be used (assault, freeway, space_invaders)
- mode specifies whether the model will be trained or evaluated (train, evaluate)
- episodes specifies how many episodes of the game the model will be trained or evaluated for
- file_name specifies what file the model will be saved to or loaded from

Example commands
```
training example:
$ python main.py assault train --episodes 1000 --file_name test_model.pth

evaluating example:
$ python main.py freeway evaluate --episodes 30 --file_name ./trained_weights/freeway_1T.pth
```
