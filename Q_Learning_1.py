# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:56:48 2020

@author: Allen
"""

import gym
import atari_py
import numpy as np
import random
import math
env = gym.make('LunarLander-v2')
print(env.action_space)
print(env.observation_space.low, env.observation_space.high)
print(env.action_space.n)
##################################################################
#OS is "observation space"
#Here we divide the observation space into smaller buckets to make
#tabular Q-learning feasible
#
##################################################################
DISCRETE_OS_SIZE = [3,3,6,6] #* len(env.observation_space.low)
visited = np.zeros(DISCRETE_OS_SIZE)
print("DS")
print(DISCRETE_OS_SIZE)
# print("High: " + str(env.observation_space.high))
# print("Low: " + str(env.observation_space.low))
# observed_adjusted_low = np.copy(env.observation_space.low)
# observed_adjusted_low[3] = -2.25
observed_adjusted_low = np.array([-2.5, -4.4, -0.31, -3.9])
# observed_adjusted_high = np.copy(env.observation_space.high)
# observed_adjusted_high[3] = 3.6
observed_adjusted_high = np.array([2.5, 4.6, 0.31, 3.95])
# print((env.observation_space.high.astype('float64') - env.observation_space.low.astype('float64')))
#discrete_os_win_size = (env.observation_space.high.astype('float64') - env.observation_space.low.astype('float64')) / DISCRETE_OS_SIZE
discrete_os_win_size = (observed_adjusted_high - observed_adjusted_low) / DISCRETE_OS_SIZE
print(discrete_os_win_size)
observation = env.reset()
done = False

class Q_Agent:
    def __init__(self, env, epsilon, discount, step_size):
        self.env = env
        self.epsilon = epsilon
        self.epsilon0 = epsilon
        self.discount = discount
        self.step_size = step_size
        self.step_size0 = step_size
        self.q = np.random.uniform(low=200, high = 300, size = (DISCRETE_OS_SIZE + [env.action_space.n]))
        self.EPISODES = 100000
        
    def agent_start(self):
        """This chooses the very first action according to an epsilon-greedy policy"""
        observation = self.env.reset()
        observation_index = tuple(self.bucket(observation))
        visited[observation_index] += 1
        #print(observation)
        current_q = self.q[observation_index]
        #print(current_q)
        r = random.random()
        if (r < self.epsilon):
            action = random.randint(0, self.env.action_space.n-1)
        else:
            action = self.argmax(current_q)
        self.prev_observation = observation_index
        self.prev_action = action
        observation, reward, done, _ = self.env.step(action)
        return observation, reward, done, _
        
    def agent_step(self, observation, reward, done):
        """This is exactly the same as agent_start, except we update the q table"""
        observation_index = tuple(self.bucket(observation))
        visited[observation_index] += 1
        #print("obs index: ", observation_index)
        current_q = self.q[observation_index]
        r = random.random()
        #print(r, self.epsilon)
        if (r < self.epsilon):
            action = random.randint(0, self.env.action_space.n-1)
            #print("random")
        else:
            action = self.argmax(current_q)
        self.q[self.prev_observation][self.prev_action] = self.q[self.prev_observation][self.prev_action] + self.step_size * (reward + self.discount*self.q[observation_index][self.argmax(current_q)] - self.q[self.prev_observation][self.prev_action])
        self.prev_observation = observation_index
        self.prev_action = action
        observation, reward, done, _ = self.env.step(action)
        return observation, reward, done, _
    

    def agent_end(self, observation, reward):
        """Updates the q table when done = True"""
        self.q[self.prev_observation][self.prev_action] = self.q[self.prev_observation][self.prev_action] + self.step_size * (reward - self.q[self.prev_observation][self.prev_action])
        
    def bucket(self, observation):
        """Convert the state into its smaller discrete size
        TO DO: Update this to take bucket sizes as a parameter"""
        return np.floor((observation-observed_adjusted_low)/discrete_os_win_size).astype(np.int32)
    
    def argmax(self, vals):
        """Gives the index in a list with the largest value. Ties are broken randomly"""
        current_best_val = float("-inf")
        indices = []
        
        for i in range(len(vals)):
            if (vals[i] >= current_best_val):
                if (vals[i] != current_best_val):
                    indices = []
                indices.append(i)
                current_best_val = vals[i]
        return random.choice(indices)
    
    def epsilon_decay(self, min_epsilon):
        #self.epsilon -= (self.epsilon0/float(self.EPISODES) - min_epsilon/float(self.EPISODES))
        self.epsilon = self.epsilon * 2**(-1/float(20000))
        
    def decay_learning_rate(self, min_learning_rate):
        self.step_size -= (self.step_size0/float(self.EPISODES) - min_learning_rate/float(self.EPISODES))
    
agent = Q_Agent(env, 0.4, 1.0, 0.5)
obs, rew, done, _ = agent.agent_start()
init_q = np.copy(agent.q)
SHOW = 1000

#############
#Set the appropriate experiment to true and all others to false
cartPole    = True
mountainCar = False
count_of_successes_cartpole = 0
rews = [0]*100
print(rews)
lowest, highest = [float('inf')]*4, [float('-inf')]*4
# while not done:
#     obs, rew, done, _ = agent.agent_step(obs, rew, done)
#     #print(rew)
#     #if(cartPole): rew_count += rew
#     #print(rew, obs)
#     #if (ep % SHOW == 0):
#     env.render()

# for ep in range(agent.EPISODES):
#     if (ep % SHOW == 0): 
#         print(ep)
#         print("Epsilon: " ,agent.epsilon)
#         print("Learning Rate: ", agent.step_size)
#         print(sum(rews))
#     rew_count = 0
#     while not done:
#         obs, rew, done, _ = agent.agent_step(obs, rew, done)
#         #print(rew)
#         if(cartPole): rew_count += rew
#         #print(rew, obs)
#         if (ep % SHOW == 0):
#             env.render()
#         for i in range(len(obs)):
#             if(obs[i] < lowest[i]):
#                 lowest[i] = obs[i]
#             if(obs[i] > highest[i]):
#                 highest[i] = obs[i]
#     agent.agent_end(obs, rew)
#     if(mountainCar):
#         if(obs[0] >= agent.env.goal_position):
#             print("Success! Episode {}".format(ep))
#     if(cartPole):
#         rews[ep % 100] = rew_count
#         if(sum(rews)/float(len(rews)) >= 195):
#             print("Success! Episode {}".format(ep))
#             count_of_successes_cartpole += 1
#         else:
#             count_of_successes_cartpole = 0
#     agent.epsilon_decay(0.08)
#     agent.decay_learning_rate(0.02)
#     obs, rew, done, _ = agent.agent_start()

# print("Observed Lowest: " + str(lowest))
# print("Observed Highest: " + str(highest))

# print(visited)
# print(rews)
# print(sum(rews))
env.close()
    
