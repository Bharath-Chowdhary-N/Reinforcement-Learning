# Deep Q Network
import numpy as np
import gymnasium as gym
import imageio
from IPython.display import Image
from gymnasium.utils import seeding

import torch
import torch.nn as nn
import torch.nn.functional as F

class  nn_model(nn.Module):
      def __init__(self,env):
          super(Model,self).__init__()
          self.env = env
          self.n_actions = env.action_space.n
          self.input_dim = env.observation_space.shape[0] # n_states are passed on to as input dim
      def forward(self, state):
          layer1 = self.nn.linear(self.input_dim,64)
          layer2 = self.nn.linear(64,32)
          layer3 = self.nn.linear(32,self.n_actions)
          activation1 = F.relu(layer1(state))
          activation2 = F.relu(layer2(activation1))
          output = layer3(activation2)
          return output

class DQN():
    def __init__(self) -> None:
        self.load_hyperparams()
        self.env = self.create_env()

    def load_hyperparams(self):
        self.gamma = 0.99
        self.n_episodes = 1000
        self.epsilon = 1
        self.minibath_size = 64
    
    def create_env(self):
        self.env = gym.make("LunarLander-v2")
        self.env.seed(1)
        return self.env
    
    def train(self):
        env = self.env
        model = nn_model(env)
        pass # Train the model


          
          
