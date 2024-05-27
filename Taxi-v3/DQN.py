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
      def __init__(self,input_dim, hidden_node, output_dim):
          super(nn_model,self).__init__()
          self.layer1 = nn.Linear(input_dim,hidden_node)
          self.layer2 = nn.Linear(hidden_node,output_dim)
      def forward(self, state):
          activation1 = F.relu(self.layer1(state))
          output = self.layer2(activation1)
          return output

class DQN():
    def __init__(self) -> None:
        
        self.env = self.create_env()
        self.load_hyperparams()
        self.create_policy_and_target_network()
        self.train()

    def load_hyperparams(self):
        self.gamma = 0.9
        self.n_episodes = 2000
        self.epsilon = 1
        self.mini_batch_size = 32
        self.max_memory_size = 1000   #check mini_batch_size ratio over max_memory_size (check literature)
        self.replay_memory=[]
        self.epsilon_history = []
        self.loss_fn = nn.MSELoss()
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
    
    def create_policy_and_target_network(self):
        self.policy_network = nn_model(input_dim=self.num_states, hidden_node=self.num_states, output_dim=self.num_actions)
        self.target_network = nn_model(input_dim=self.num_states, hidden_node=self.num_states, output_dim=self.num_actions)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        print("---------------------------------------------Policy and Target Network created---------------------------------------------------")
    def train(self):
        step_count = 0
        for ite in range(self.n_episodes):
            state = self.env.reset()[0]
            done  = False
            trunc = False
            while(not done and not trunc):
                if np.random.rand() < self.epsilon: 
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.policy_network(torch.from_numpy(np.reshape(state,[1,self.num_states])))

                next_state, reward, done, trunc, info = self.env.step(action)        

                if len(self.replay_memory) < self.max_memory_size:
                    self.replay_memory.append({"state":state, "action":action, "reward":reward, "next_state":next_state, "done":done})
                else:
                    self.replay_memory.pop(0)
                
                #print(len(self.replay_memory))
            if reward > 0:
               self.rewards_per_episode[ite] = reward 
                
            if len(self.replay_memory)>self.mini_batch_size and np.sum(self.rewards_per_episode)>0:
               minibatch = np.random.choice(self.replay_memory, self.mini_batch_size, replace=True) 
               self.optimize(minibatch)
    def optimize(self, minibatch):
        state_list      =      torch.from_numpy(np.array(list(map(lambda x: x['state'], minibatch))))
        action_list     =      np.array(list(map(lambda x: x['action'], minibatch)))
        reward_list     =      np.array(list(map(lambda x: x['reward'], minibatch)))
        next_state_list =      torch.from_numpy(np.array(list(map(lambda x: x['next_state'], minibatch))))
        done_list       =      np.array(list(map(lambda x: x['done'], minibatch)))
        current_q_list = []
        target_q_list = []
        for ite in range(self.mini_batch_size):
            
            if done_list[ite]:
                target = torch.FloatTensor([reward_list[ite]])
            else:
                 
                with torch.no_grad():
                    target = torch.FloatTensor( reward_list[ite] + self.gamma* self.target_network(torch.from_numpy(np.reshape(state_list[ite],[1,self.num_states]))).max()
                                               )
            
            current_q = self.policy_network(torch.from_numpy(np.reshape(state_list[ite],[1,self.num_states])))
            current_q_list.append(current_q)

            target_q = self.target_network(torch.from_numpy(np.reshape(state_list[ite],[1,self.num_states])))
            target_q[action_list[ite]] = target
            target_q_list.append(target_q)
        
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
'''
    def replay(self, replay_memory):
        
        minibatch = np.random.choice(replay_memory, self.mini_batch_size, replace=True)

        #print(minibatch['state'])
        #for i, sample in enumerate(minibatch):
        #       print(f"Sample {i} state shape: {np.array(sample['state']).shape}")

        
        state_list      =      torch.from_numpy(np.array(list(map(lambda x: x['state'], minibatch))))
        action_list     =      np.array(list(map(lambda x: x['action'], minibatch)))
        reward_list     =      np.array(list(map(lambda x: x['reward'], minibatch)))
        next_state_list =      torch.from_numpy(np.array(list(map(lambda x: x['next_state'], minibatch))))
        done_list       =      np.array(list(map(lambda x: x['done'], minibatch)))

        q_values_next_state        =      self.model(next_state_list)

        q_values_current_state     =      self.model(state_list)

        q_values_update            =      self.model(state_list)

        for ite,(state,action,reward,q_values_next_state, done) in enumerate(zip(state_list,action_list,reward_list,q_values_next_state, done_list)): 
            if not done:  
                target = reward + self.gamma * np.max(q_values_next_state.detach().numpy())
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
        
        self.episode_rewards = []
        for episode in range(self.n_episodes):
            state = env.reset()
            done = False
            sum_reward = 0
            state= state[0]
            self.len_state = len(state)
            while not done:
                #print("{}".format(state[0]))
                q_action = self.model(torch.from_numpy(np.reshape(state,[1,self.len_state]))) #output from DQN model
                
                #print("-------------------------------------------------This has passed successfully---------------------------------------------")
                #print(state)
                #take action according to epsilon-greedy policy
                if np.random.rand() < self.epsilon: 
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_action.detach().numpy())
                #print(env.step(action))
                # execute the action 
                next_state, reward, done, info, _ = env.step(action)
                
                # add reward to the sum
                sum_reward += reward
                
                #print("this is before: {} {}".format(state, next_state))
                # add the transition to the replay memory
                if len(self.replay_memory) < self.max_memory_size:
                    self.replay_memory.append({"state":state, "action":action, "reward":reward, "next_state":next_state, "done":done})
                else:
                    self.replay_memory.pop(0)
                    
                #add replay functionality here
                self.model  = self.replay(self.replay_memory)

                #update state
                state = next_state                
                
                if self.epsilon > 0.01:
                    self.epsilon -= 0.01
            
            print(f"Episode: {episode}, Reward: {sum_reward}")
            self.episode_rewards.append(sum_reward) 
        #with open("file.txt", "w") as output:
            np.save("file.npy", np.array(self.episode_rewards))
'''

if __name__ == "__main__":
    dqn = DQN()
