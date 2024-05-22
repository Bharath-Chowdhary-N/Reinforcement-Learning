# Deep Q Network
import numpy as np
import gymnasium as gym
import imageio
from IPython.display import Image
from gymnasium.utils import seeding

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class  nn_model(nn.Module):
      def __init__(self,env):
          super(nn_model,self).__init__()
          self.env = env
          self.n_actions = env.action_space.n
          self.input_dim = env.observation_space.shape[0] # n_states are passed on to as input dim
          self.layer1 = nn.Linear(self.input_dim,64)
          self.layer2 = nn.Linear(64,32)
          self.layer3 = nn.Linear(32,self.n_actions)
      def forward(self, state):
          activation1 = F.relu(self.layer1(state))
          activation2 = F.relu(self.layer2(activation1))
          output = self.layer3(activation2)
          return output

class DQN():
    def __init__(self) -> None:
        
        self.env = self.create_env()
        self.load_hyperparams()
        self.train()

    def load_hyperparams(self):
        self.gamma = 0.99
        self.n_episodes = 1000
        self.epsilon = 1
        self.mini_batch_size = 64
        self.max_memory_size = 100000   #check mini_batch_size ratio over max_memory_size (check literature)
        self.replay_memory=[]
        self.model = nn_model(self.env)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def create_env(self):
        self.env = gym.make("LunarLander-v2")
        #self.env.seed(1)
        return self.env
    
    def replay(self, replay_memory):
        
        minibatch = np.random.choice(replay_memory, self.mini_batch_size, replace=True)
        
        state_list      =      np.array(list(map(lambda x: x['state'], minibatch)))
        action_list     =      np.array(list(map(lambda x: x['action'], minibatch)))
        reward_list     =      np.array(list(map(lambda x: x['reward'], minibatch)))
        next_state_list =      np.array(list(map(lambda x: x['next_state'], minibatch)))
        done_list       =      np.array(list(map(lambda x: x['done'], minibatch)))

        q_values_next_state        =      self.model.predict(next_state_list)

        q_values_current_state     =      self.model.predict(state_list)

        q_values_update            =      self.model.predict(state_list)

        for ite,(state,action,reward,q_values_next_state, done) in enumerate(zip(state_list,action_list,reward_list,q_values_next_state, done_list)): 
            if not done:  
                target = reward + self.gamma * np.max(q_values_next_state)
            else:
                target = reward

            q_values_update[ite][action] = target
        
        loss = self.loss_fn(q_values_update, q_values_current_state)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 

        return self.model

    
    def train(self):
        env = self.env
        
        episode_rewards = []
        for episode in range(self.n_episodes):
            state = env.reset()
            done = False
            sum_reward = 0
            self.len_state = len(state[0])
            while not done:
                print("{}".format(state[0]))
                q_action = self.model(torch.from_numpy(np.reshape(state[0],[1,self.len_state]))) #output from DQN model
                
                #take action according to epsilon-greedy policy
                if np.random.rand() < self.epsilon: 
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_action.detach().numpy())
                print(env.step(action))
                # execute the action 
                next_state, reward, done, info, _ = env.step(action)
                
                # add reward to the sum
                sum_reward += reward
                
                # add the transition to the replay memory
                if len(self.replay_memory) < self.max_memory_size:
                    self.replay_memory.append({"state":state, "action":action, "reward":reward, "next_state":next_state, "done":done})
                else:
                    self.replay_memory.pop(0)
                    
                #add replay functionality here
                self.model  = self.replay(self.replay_memory)

                #update state
                state = next_state                
                
                if epsilon > 0.01:
                    epsilon -= 0.01
            
            print(f"Episode: {episode}, Reward: {sum_reward}")
            episode_rewards.append(sum_reward) 



if __name__ == "__main__":
    DQN()
