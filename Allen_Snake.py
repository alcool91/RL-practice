# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:34:55 2020

@author: Allen
"""
#######################################################
#  Followed a PyGame tutorial at the following address
#  https://www.youtube.com/watch?v=FfWpgLFMI7w
#


from pygame.locals import *
import random
import pygame
import time
import numpy as np
import pickle

#####################################################
# Constants
#
SCREEN_WIDTH  = 880
SCREEN_HEIGHT = 880
APPLE_REWARD  = 50
COLLISION_PENALTY = 750
TURN_PENALTY = 1

state_space = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
action_space = np.array([0,1,2,3,4]).astype(np.int32)

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
logo = pygame.image.load('funny_face.jpg')
snakeImg = pygame.image.load('block.jpg')
appleImg  = pygame.image.load('apple.jpg')
pygame.display.set_icon(logo)

pygame.display.set_caption("Allen's Homemade Snake Game")


class Player:
    direction = 0
    def __init__(self, start_x, start_y):
        self.x = []
        self.y = []
        self.x.append(start_x)
        self.x.append(start_x-44)
        self.x.append(start_x-88)
        self.y.append(start_y)
        self.y.append(start_y)
        self.y.append(start_y)
        
    def reset(self, start_x, start_y):
        #print(self.x, self.y)
        #print("Resetting Snake")
        self.x = []
        self.y = []
        self.direction = 0
        self.x.append(start_x)
        self.x.append(start_x-44)
        self.x.append(start_x-88)
        self.y.append(start_y)
        self.y.append(start_y)
        self.y.append(start_y)
        #print(self.x, self.y)
        
    def move(self, direction):
        self.last_tail_x = self.x[-1]; self.last_tail_y = self.y[-1]
        for i in range(len(self.x)-1,0,-1):
                self.x[i] = self.x[i-1]
                self.y[i] = self.y[i-1]
        if direction == 0:
            self.x[0] = (self.x[0] + 44) % SCREEN_WIDTH
        elif direction == 1:
            self.y[0] = (self.y[0] + 44) % SCREEN_HEIGHT
        elif direction == 2:
            self.x[0] = (self.x[0] - 44) % SCREEN_WIDTH
        elif direction == 3:
            self.y[0] = (self.y[0] - 44) % SCREEN_HEIGHT
        
            
    def grow(self):
        self.x.append(self.last_tail_x)
        self.y.append(self.last_tail_y)
        
    def set_direction(self, direction):
        # if (self.direction != direction):
        #     print("Direction Changed")
        self.direction = direction
        
    def is_collision(self, move_projection_x = 0, move_projection_y = 0):
        for i in range(len(self.x)-1, 0, -1):
            if (self.x[0] + move_projection_x) % SCREEN_WIDTH == self.x[i] and (self.y[0] + move_projection_y) % SCREEN_HEIGHT == self.y[i]:
                return True
        return False
    
    def get_dist(self, other):
        return (self.x-other.x) + (self.y-other.y)
    
    def get_dist_vector(self, other):
        # vec = [(self.x[0]-other.x)//44, (self.y[0]-other.y)//44]
        # if vec[0] < 0:
        #     vec[0] = vec[0] + 20
        # if vec[1] < 0:
        #     vec[1] = vec[1] + 20
        vec = [(self.x[0]-other.x)//44, (self.y[0]-other.y)//44]
        if vec[0] < 0:
            vec[0] = 0
        elif vec[0] == 0:
            vec[0] = 1
        elif vec[0] > 0:
            vec[0] = 2
        if vec[1] < 0:
            vec[1] = 0
        elif vec[1] == 0:
            vec[1] = 1
        elif vec[1] > 0:
            vec[1] = 2
        return tuple(vec)


class Food:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        print("Apple")
        print(self.x, self.y)
        
    def relocate(self):
        self.x = 44*random.randint(0,19)
        self.y = 44*random.randint(0,19)
        # print("Apple")
        # print(self.x, self.y)
        screen.blit(appleImg, (self.x, self.y))


class Q_Snake:
    
    def __init__(self, epsilon, learning_rate, discount, new_q = True):
        self.mySnake = Player(352, 484)
        #self.q = np.random.uniform(low=0, high=1, size=(5, 40, 40, 5))
        self.q = np.random.uniform(low=0, high=1, size=(3, 3, 3, 3, 3, 3, 5))
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon0 = epsilon
        self.EPISODES = 1000001
        if not new_q:
            qt = open('qtable.pickle', 'rb')
            self.q = pickle.load(qt, encoding='bytes')
            qt.close()
            print("loaded qtable...")
        
    def agentStart(self, obs):
        if (random.random() < self.epsilon):
            action = random.randint(0,4)
        else:
            action = np.argmax(self.q[obs[0]][obs[1]][obs[2]][obs[3]][obs[4]][obs[5]])
            
        self.last_obs = obs
        self.last_action = action
        return action
        
    def takeAction(self, obs, rew):
        if (random.random() < self.epsilon):
            action = random.randint(0,4)
        else:
            action = np.argmax(self.q[obs[0]][obs[1]][obs[2]][obs[3]][obs[4]][obs[5]])
            
        self.q[self.last_obs[0]][self.last_obs[1]][self.last_obs[2]][self.last_obs[3]][self.last_obs[4]][self.last_obs[5]][self.last_action] = self.q[self.last_obs[0]][self.last_obs[1]][self.last_obs[2]][self.last_obs[3]][self.last_obs[4]][self.last_obs[5]][self.last_action] + self.learning_rate*(rew + self.discount*(np.max(self.q[obs[0]][obs[1]][obs[2]][obs[3]][obs[4]][obs[5]])) - self.q[self.last_obs[0]][self.last_obs[1]][self.last_obs[2]][self.last_obs[3]][self.last_obs[4]][self.last_obs[5]][self.last_action])
        self.last_obs = obs
        self.last_action = action
        return action
        
    def agentEnd(self, obs, rew):
        self.q[self.last_obs[0]][self.last_obs[1]][self.last_obs[2]][self.last_obs[3]][self.last_obs[4]][self.last_obs[5]][self.last_action] = self.q[self.last_obs[0]][self.last_obs[1]][self.last_obs[2]][self.last_obs[3]][self.last_obs[4]][self.last_obs[5]][self.last_action] + self.learning_rate*(rew  - self.q[self.last_obs[0]][self.last_obs[1]][self.last_obs[2]][self.last_obs[3]][self.last_obs[4]][self.last_obs[5]][self.last_action])
        #print("reset snake?")
        self.mySnake.reset(352, 484)
        
    def epsilon_decay(self, min_epsilon):
        #self.epsilon -= (self.epsilon0/float(self.EPISODES) - min_epsilon/float(self.EPISODES))
        self.epsilon = self.epsilon * 2**(-1/float(100000))
        

running = True
#snake = Player(352, 484)
apple = Food(44*random.randint(0,19), 44*random.randint(0,19))

# while(running):
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#     screen.fill((0,0,0))
    
#     pygame.event.pump()
#     keys = pygame.key.get_pressed()
    
#     if keys[K_RIGHT]:
#         snake.set_direction(0)
#     elif keys[K_LEFT]:
#         snake.set_direction(2)
#     elif keys[K_DOWN]:
#         snake.set_direction(1)
#     elif keys[K_UP]:
#         snake.set_direction(3)
#     snake.move(snake.direction)
#     if snake.is_collision():
#         print("COLLISION! Game Over!!")
#         running = False
#         continue
#     for i in range(len(snake.x)):
#         screen.blit(snakeImg, (snake.x[i], snake.y[i]))
    
#     #screen.blit(snakeImg, (snake.x, snake.y))
#     screen.blit(appleImg, (apple.x, apple.y))
#     #snake.move(random.randint(0,4))
#     if(snake.x[0] == apple.x and snake.y[0] == apple.y):
#         apple.relocate()
#         snake.grow()
#     pygame.display.update()
#     time.sleep(0.10)
#     print(snake.x, snake.y)


learner = Q_Snake(1, 0.06, 0.995, True)
f = open('qtable.pickle', 'wb')
for j in range(learner.EPISODES):
    show = False
    if j % 1000 == 0 and j != 0: show = True
    print("j=",j)
    running = True
    dist_vect = learner.mySnake.get_dist_vector(apple)
    obs = [0, dist_vect[0], dist_vect[1],0,0,0]
    print(obs)
    act = learner.agentStart(obs)
    rews = 0
    
    while(running):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
        if show: screen.fill((0,0,0))
        obs = [0,0,0,0,0,0]
        # pygame.event.pump()
        # keys = pygame.key.get_pressed()
        # if j > learner.EPISODES // 4:
        #     TURN_PENALTY = 1
        # else:
        #     TURN_PENALTY = -(len(learner.mySnake.x)-3)
        rew = -TURN_PENALTY
        if act == 0:
            learner.mySnake.set_direction(0)
        elif act == 2:
            learner.mySnake.set_direction(2)
        elif act == 1:
            learner.mySnake.set_direction(1)
        elif act == 3:
            learner.mySnake.set_direction(3)
        learner.mySnake.move(learner.mySnake.direction)
        if learner.mySnake.is_collision(44, 0):
            obs[0] = 1
        elif learner.mySnake.is_collision(88,0):
            obs[0] = 2
        if learner.mySnake.is_collision(-44, 0):
            obs[3] = 1
        elif learner.mySnake.is_collision(-88,0):
            obs[3] = 2
        if learner.mySnake.is_collision(0, 44):
            obs[4] = 1
        elif learner.mySnake.is_collision(0,88):
            obs[4] = 2
        if learner.mySnake.is_collision(0, -44):
            obs[5] = 1
        elif learner.mySnake.is_collision(0,-88):
            obs[5] = 2
        
            
        dist_vect = learner.mySnake.get_dist_vector(apple)
        obs[1] = dist_vect[0]
        obs[2] = dist_vect[1]
        if learner.mySnake.is_collision():
            #print("COLLISION! Game Over!!")
            rew = -COLLISION_PENALTY
            rews += rew
            print(rews)
            learner.agentEnd(obs, rew)
            running = False
            learner.epsilon_decay(0.1)
            print(learner.epsilon)
            continue
        if show:
            for i in range(len(learner.mySnake.x)):
                screen.blit(snakeImg, (learner.mySnake.x[i], learner.mySnake.y[i]))
        
        #screen.blit(snakeImg, (snake.x, snake.y))
        if show: screen.blit(appleImg, (apple.x, apple.y))
        #snake.move(random.randint(0,4))
        if(learner.mySnake.x[0] == apple.x and learner.mySnake.y[0] == apple.y):
            rew = APPLE_REWARD
            apple.relocate()
            learner.mySnake.grow()
        if show: pygame.display.update()
        rews += rew
        act = learner.takeAction(obs, rew)
        if show: time.sleep(0.11)
        #print(learner.mySnake.x, learner.mySnake.y)
        #print("i=", i)
         
pickle.dump(learner.q, f)
f.close()
pygame.quit()