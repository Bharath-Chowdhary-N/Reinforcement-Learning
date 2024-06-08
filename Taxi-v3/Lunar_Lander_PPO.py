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

class  Critic_Model(nn.Module):
      """
      This is for Value function update. Loss is usually MSE loss.
      """
      def __init__(self):
          super(Critic_Model,self).__init__()
          critic_layer_dim = [512,256,64,1]
          self.layer1 = nn.Linear(critic_layer_dim[0], critic_layer_dim[1])
          self.layer2 = nn.Linear(critic_layer_dim[1], critic_layer_dim[2])
          self.layer3 = nn.Linear(critic_layer_dim[2],critic_layer_dim[3])
      def forward(self, state):
          activation1 = F.relu(self.layer1(state))
          #activation2 = F.relu(activation1)
          activation2 = F.relu(self.layer2(activation1))
          output      = self.layer3(activation2)
          return output


class PPO():
    def __init__(self) -> None:
        
        self.env = self.create_env()
        self.load_hyperparams()
        self.create_actor_and_critic_network()
        #self.create_policy_and_target_network()
        #self.train()

    def load_hyperparams(self):
        self.gamma = 0.9
        self.n_episodes = 2000
        self.epsilon = 1
        self.mini_batch_size = 32
        self.max_memory_size = 1000   #check mini_batch_size ratio over max_memory_size (check literature)
        self.replay_memory=[]
        self.epsilon_history = []
        self.loss_fn = nn.MSELoss()
        self.learning_rate=0.0001
        self.network_sync_rate = 10
        #self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.rewards_per_episode = np.zeros(self.n_episodes)
    def create_env(self):
        self.env = gym.make("LunarLander-v2")
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        print("-------------------------------------------------------\n  Welcome to Lunar lander! . Num states : {} , Num actions : {}       \n-----------------------------------------------------".format(self.num_states, self.num_actions))
        #self.env.seed(1)
        return self.env
    
    def create_actor_and_critic_network(self):
        self.critic_network = Critic_Model()
        print("---------------------------------------------Critic Network created---------------------------------------------------")
    
    def create_policy_and_target_network(self):
        self.policy_network = nn_model(input_dim=self.num_states, hidden_node=self.num_states, output_dim=self.num_actions)
        self.target_network = nn_model(input_dim=self.num_states, hidden_node=self.num_states, output_dim=self.num_actions)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        print("---------------------------------------------Policy and Target Network created---------------------------------------------------")
    def train(self):
        step_count = 0
        for ite in range(self.n_episodes):
            state = self.env.reset()[0]
            done  = False
            trunc = False
            while(not done and not trunc):
                if np.random.rand() < self.epsilon: 
                #if False:    
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.policy_network(torch.from_numpy(np.reshape(state,[1,self.num_states]))).argmax().cpu().detach().numpy()

                next_state, reward, done, trunc, info = self.env.step(action)        

                if len(self.replay_memory) < self.max_memory_size:
                    self.replay_memory.append({"state":state, "action":action, "reward":reward, "next_state":next_state, "done":done})
                else:
                    self.replay_memory.pop(0)
                state = next_state
                step_count+=1
                #print(len(self.replay_memory))
            if reward > 0:
               self.rewards_per_episode[ite] = reward 
                
            if len(self.replay_memory)>self.mini_batch_size and np.sum(self.rewards_per_episode)>0:
               minibatch = np.random.choice(self.replay_memory, self.mini_batch_size, replace=True) 
               self.optimize(minibatch)

               self.epsilon = max(self.epsilon -1/self.n_episodes, 0)
               self.epsilon_history.append(self.epsilon)

               if step_count > self.network_sync_rate:
                   self.target_network.load_state_dict(self.policy_network.state_dict())
                   step_count=0
            if ite%100==0:
                print("----------------------{}: has been processed------------------------".format(ite))
        torch.save(self.policy_network.state_dict(), "lunar_lander.pt")
if __name__ == "__main__":
    ppo = PPO()