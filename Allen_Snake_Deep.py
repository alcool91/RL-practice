# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:16:17 2020

@author: Allen
"""

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
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
from statistics import mean

#####################################################
# Constants
#
SCREEN_WIDTH  = 880
SCREEN_HEIGHT = 880
APPLE_REWARD  = 420
COLLISION_PENALTY = 10
TURN_PENALTY = 1
BLOCK_SIZE = 44

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
        start_x = random.randint(2,17)*BLOCK_SIZE
        start_y = random.randint(2,17)*BLOCK_SIZE
        start_dir = random.randint(0,3)
        self.direction = start_dir
        self.x = []
        self.y = []
        self.x.append(start_x)
        self.y.append(start_y)
        self.debug_x = self.x.copy()
        
        if start_dir == 0:
            #RIGHT
            self.x.append(start_x-BLOCK_SIZE)
            self.x.append(start_x-2*BLOCK_SIZE)
            
            self.y.append(start_y)
            self.y.append(start_y)
            
        elif start_dir == 1:
            #DOWN
            self.x.append(start_x)
            self.x.append(start_x)
            
            self.y.append(start_y-BLOCK_SIZE)
            self.y.append(start_y-2*BLOCK_SIZE)
            
        elif start_dir == 2:
            #LEFT
            self.x.append(start_x+BLOCK_SIZE)
            self.x.append(start_x+2*BLOCK_SIZE)
            
            self.y.append(start_y)
            self.y.append(start_y)
            
        elif start_dir == 3:
            #UP
            self.x.append(start_x)
            self.x.append(start_x)
            
            self.y.append(start_y+BLOCK_SIZE)
            self.y.append(start_y+2*BLOCK_SIZE)
            
        self.debug_x = self.x.copy()
        self.debug_y = self.y.copy()
            
        
        
        
    def print_debug_info(self):
        print([x//44 for x in self.x], [y//44 for y in self.y])
        print("\n Starting Coordinate Arrays \n")
        print([x//44 for x in self.debug_x], [y//44 for y in self.debug_y])
        print("\n Direction \n")
        print(self.direction)
        
    def reset(self, start_x, start_y):
        start_x = random.randint(2,17)*BLOCK_SIZE
        start_y = random.randint(2,17)*BLOCK_SIZE
        start_dir = random.randint(0,3)
        self.direction = start_dir
        self.x = []
        self.y = []
        self.x.append(start_x)
        self.y.append(start_y)
        
        if start_dir == 0:
            #RIGHT
            self.x.append(start_x-BLOCK_SIZE)
            self.x.append(start_x-2*BLOCK_SIZE)
            
            self.y.append(start_y)
            self.y.append(start_y)
            
        elif start_dir == 1:
            #DOWN
            self.x.append(start_x)
            self.x.append(start_x)
            
            self.y.append(start_y-BLOCK_SIZE)
            self.y.append(start_y-2*BLOCK_SIZE)
            
        elif start_dir == 2:
            #LEFT
            self.x.append(start_x+BLOCK_SIZE)
            self.x.append(start_x+2*BLOCK_SIZE)
            
            self.y.append(start_y)
            self.y.append(start_y)
            
        elif start_dir == 3:
            #UP
            self.x.append(start_x)
            self.x.append(start_x)
            
            self.y.append(start_y+BLOCK_SIZE)
            self.y.append(start_y+2*BLOCK_SIZE)
            
        self.debug_x = self.x.copy()
        self.debug_y = self.y.copy()
            

        
    def move(self, direction):
        self.last_tail_x = self.x[-1]; self.last_tail_y = self.y[-1]
        for i in range(len(self.x)-1,0,-1):
                self.x[i] = self.x[i-1]
                self.y[i] = self.y[i-1]
        if direction == 0:
            self.x[0] = (self.x[0] + BLOCK_SIZE) % SCREEN_WIDTH
        elif direction == 1:
            self.y[0] = (self.y[0] + BLOCK_SIZE) % SCREEN_HEIGHT
        elif direction == 2:
            self.x[0] = (self.x[0] - BLOCK_SIZE) % SCREEN_WIDTH
        elif direction == 3:
            self.y[0] = (self.y[0] - BLOCK_SIZE) % SCREEN_HEIGHT
        
            
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
                print(self.x[0]//44, self.y[0]//44, i, len(self.x))
                return True
        return False
    
    def get_dist(self, other):
        return (self.x[0]//BLOCK_SIZE-other.x//BLOCK_SIZE) + (self.y[0]//BLOCK_SIZE-other.y//BLOCK_SIZE)
    
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

    def get_coords(self):
        return [np.array([self.x[i]/44, self.y[i]/44]).astype(int) for i in range(len(self.x))]

class Food:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        print("Apple")
        print(self.x//44, self.y//44)
        
    def relocate(self):
        self.x = 44*random.randint(0,19)
        self.y = 44*random.randint(0,19)
        # print("Apple")
        # print(self.x, self.y)
        screen.blit(appleImg, (self.x, self.y))

class Board:
    
    def __init__(self, snake_coords, food_coords):
        print(snake_coords, snake_coords[0][0], type(snake_coords[0][0]))
        self.board = np.zeros((20,20))
        self.board[snake_coords[0][1], snake_coords[0][0]] = 1
        for i in range(1, len(snake_coords)):
            self.board[snake_coords[i][1], snake_coords[i][0]] = 2
        for i in range(len(food_coords)):
            self.board[food_coords[i, 1], food_coords[i, 0]] = 3
    
    def update(self, snake_coords, food_coords):
        self.board = np.zeros((20,20))
        self.board[snake_coords[0][1], snake_coords[0][0]] = 1
        for i in range(1, len(snake_coords)):
            self.board[snake_coords[i][1], snake_coords[i][0]] = 2
        for i in range(len(food_coords)):
            self.board[food_coords[i, 1], food_coords[i, 0]] = 3
            
    def get_board(self):
        board2 = self.board.copy()
        #board2 = np.expand_dims(board2, axis=0)
        board2 = np.expand_dims(board2, axis=-1)
        return board2
        
    
    
class Q_Snake:
    
    def __init__(self, epsilon, learning_rate, discount, new_q = True):
        self.mySnake = Player(352, 484)
        self.FILTERS = 32 
        self.CONV_WINDOW_SIZE = 3 
        self.DENSE_LAYER_NEURONS = 64 
        self.LEARNING_RATE_ADAM = 0.0001 
        self.BATCH_SIZE = 32
        self.MINIMUM_REPLAY_MEMORY = 1000
        self.directions = {
            "RIGHT": 0,
            "LEFT": 2,
            "DOWN": 1,
            "UP": 3
            }
        #self.model = self.create_model()
        
        self.replay_memory = deque(maxlen=500000)
        
        self.target_update_counter = 0
        #self.q = np.random.uniform(low=0, high=1, size=(5, 40, 40, 5))
        #self.q = np.random.uniform(low=0, high=1, size=(3, 3, 3, 3, 3, 3, 21, 5))
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.EPISODES = 100001
        self.epsilon0 = epsilon
        if not new_q:
            self.model = tf.keras.models.load_model("model2")
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())
            #self.q = pickle.load(qt, encoding='bytes')
            #qt.close()
            print("loaded model...")
        else:
            self.model = self.create_model()
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())
        
        
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(self.FILTERS, (3,3), input_shape=(20,20,1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        #model.add(Dropout(0.2))
        
        model.add(Conv2D(self.FILTERS, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        #model.add(Dropout(0.2))
        
        model.add(Flatten())
        model.add(Dense(self.DENSE_LAYER_NEURONS))
        
        model.add(Dense(4, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE_ADAM), metrics=['accuracy'])
        
        return model 
    
    def update_replay_memory(self, trans):
        self.replay_memory.append(trans)
        
    def get_qs(self, state):
        #print(state/4, state.shape)
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/4)[0]
    
    def train(self, terminal_state):
        if len(self.replay_memory) < self.MINIMUM_REPLAY_MEMORY:
            return
        
        minibatch = random.sample(self.replay_memory, self.BATCH_SIZE) 
        
        current_states = np.array([trans[0] for trans in minibatch])/4 
        current_qs_list = self.model.predict(current_states)
        
        new_current_states = np.array([trans[3] for trans in minibatch])
        future_qs_list = self.target_model.predict(new_current_states) 
        
        X = []
        y = []
        
        for i, (s, a, r, sp, done) in enumerate(minibatch):
            
            if not done:
                max_future_q = np.max(future_qs_list[i]) 
                new_q = r + self.discount * max_future_q 
                
            else:
                new_q = r 
                
            current_qs = current_qs_list[i] 
            current_qs[a] = new_q 
            
            X.append(s)
            y.append(current_qs) 
        self.model.fit(np.array(X)/4, np.array(y), batch_size=32, verbose=0, shuffle=False) 
        #print(np.sum(self.model.layers[0].get_weights()[0]))
        if terminal_state:
            self.target_update_counter += 1 
            
        if self.target_update_counter > 10000:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    
    def agentStart(self, obs):
        # print("Agent Start Direction: ", self.mySnake.direction)
        possibilities = [0,1,2,3]
        invalid = -1
        if self.mySnake.direction == 0:
            invalid = 2
            possibilities.remove(2)
        elif self.mySnake.direction == 1:
            invalid = 3
            possibilities.remove(3)
        elif self.mySnake.direction == 2:
            invalid = 0
            possibilities.remove(0)
        elif self.mySnake.direction == 3:
            invalid = 1
            possibilities.remove(1)
        if (random.random() < self.epsilon):
            action = random.choice(possibilities)
        else:
            valid_actions = self.get_qs(obs).copy()
            valid_actions[invalid] = -math.inf
            action = np.argmax(valid_actions)
        # print("Start possibilities: ", possibilities)
        # print(self.get_qs(obs))
        # print(self.get_qs(obs)[possibilities])
        # print(action)
        
        self.last_obs = obs
        self.last_action = action
        return action
        
    def takeAction(self, obs, rew):
        possibilities = [0,1,2,3]
        invalid = -1
        if self.mySnake.direction == 0:
            invalid = 2
            possibilities.remove(2)
        elif self.mySnake.direction == 1:
            invalid = 3
            possibilities.remove(3)
        elif self.mySnake.direction == 2:
            invalid = 0
            possibilities.remove(0)
        elif self.mySnake.direction == 3:
            invalid = 1
            possibilities.remove(1)
        if (random.random() < self.epsilon):
            action = random.choice(possibilities)
        else:
            valid_actions = self.get_qs(obs).copy()
            valid_actions[invalid] = -math.inf
            action = np.argmax(valid_actions)
            
        
        
        #print(len(self.replay_memory))
        self.train(False) 
        self.last_obs = obs
        self.last_action = action
        return action
        
    def agentEnd(self, obs, rew):
        self.train(True)
        #print("reset snake?")
        self.mySnake.reset(352, 484)
        
    def epsilon_decay(self, min_epsilon, ep):
        if ep < 5000:
            hl = 5000
        elif 5000 < ep < 50000:
            hl = 5000
        else:
            hl = 15000
        #self.epsilon -= (self.epsilon0/float(self.EPISODES) - min_epsilon/float(self.EPISODES))
        self.epsilon = max(self.epsilon * 2**(-1/float(hl)), min_epsilon)
        
debug_dirs = []
running = True
apple = Food(44*random.randint(0,19), 44*random.randint(0,19))
learner = Q_Snake(1, 0.3, 0.995, True)
game_board = Board(learner.mySnake.get_coords(), np.array([(apple.x/44, apple.y/44)]).astype(int))
prev_board = game_board.get_board().copy()
max_rew = -math.inf
#print(game_board.board)
f = open('modelweights2.pickle', 'wb')
prev_buffer = 0

SHOW_REWARDS_SUMMARY = 10
episode_rewards = deque(maxlen = SHOW_REWARDS_SUMMARY)
for j in range(learner.EPISODES):
    show = True
    #if j % 500 == 0 and j != 0: show = True
    print("Episode: ",j)
    print(learner.epsilon)
    running = True
    obs = game_board.get_board()
    #print(obs)
    act = learner.agentStart(obs)
    prev_act = -1
    rews = 0
    
    while(running):
        #time.sleep(1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
        if show: screen.fill((0,0,0))
        obs = game_board.get_board()
        #print(max((1-(j/(learner.EPISODES/4))),0)*-(38-learner.mySnake.get_dist(apple)))
        
        ###################################################################
        #For shaped rewards phased out over the first 1/4 epsisodes
        TURN_PENALTY = min((j/(learner.EPISODES/4)),1) + 0.001*max((1-(j/(learner.EPISODES/4))),0)*-(38-learner.mySnake.get_dist(apple))
        #
        
        
        # if j > learner.EPISODES // 4:
        #     TURN_PENALTY = 1
        # else:
        #     TURN_PENALTY = -(len(learner.mySnake.x)-3)
        rew = -TURN_PENALTY
        #prev_act = act
        #act = 3
        if act == 0:
            learner.mySnake.set_direction(0)
        elif act == 2:
            learner.mySnake.set_direction(2)
        elif act == 1:
            learner.mySnake.set_direction(1)
        elif act == 3:
            learner.mySnake.set_direction(3)
        learner.mySnake.move(learner.mySnake.direction)
        #print(game_board.board)
        #print(learner.mySnake.direction)

        if learner.mySnake.is_collision():
            #print("COLLISION! Game Over!!")
            rew = -TURN_PENALTY-COLLISION_PENALTY
            rews += rew
            if rews > max_rew: max_rew = rews
            print("{:.2f} {:.2f} {}".format(rews, max_rew, len(learner.replay_memory)))
            delta_buffer = len(learner.replay_memory) - prev_buffer
            prev_buffer = len(learner.replay_memory)
            print("Frames this Episode: ", delta_buffer)
            if(delta_buffer == 1):
                print("-- Suicide Debug Info --")
                print("\n Board\n")
                print(game_board.board)
                print("\n Current Direction \n")
                print(learner.mySnake.direction)
                print("\n x,y coordinate arrays \n")
                print(learner.mySnake.print_debug_info())
                print("\n previous action \n")
                print(prev_act)
                print("-- End --")
            if j % SHOW_REWARDS_SUMMARY == 0 and j != 0:
                print("--- Statistics for previous ", SHOW_REWARDS_SUMMARY, "episodes at episode ", j, ' ---')
                print("Average Reward Over ", SHOW_REWARDS_SUMMARY, "Episodes: ", mean(episode_rewards) )
                print("Maximum Reward Over ", SHOW_REWARDS_SUMMARY, "Episodes: ", max(episode_rewards) )
                print("Minimum Reward Over ", SHOW_REWARDS_SUMMARY, "Episodes: ", min(episode_rewards) )
                episode_rewards = []
            # if rews == -10:
            #     #learner.mySnake.print_debug_info()
            #     debug_dirs.append(learner.mySnake.direction)
            #     print(debug_dirs)
            learner.agentEnd(obs, rew)
            episode_rewards.append(rews)
            learner.update_replay_memory((prev_board, act, rew, game_board.get_board(), True))
            game_board.update(learner.mySnake.get_coords(), np.array([(apple.x/44, apple.y/44)]).astype(int))
            prev_board = game_board.get_board()
            apple.relocate()
            running = False
            learner.epsilon_decay(0.1, j)
            if show: print(learner.epsilon)
            continue
        if show:
            for i in range(len(learner.mySnake.x)):
                screen.blit(snakeImg, (learner.mySnake.x[i], learner.mySnake.y[i]))
                #print(game_board.board)
        
        #screen.blit(snakeImg, (snake.x, snake.y))
        if show: screen.blit(appleImg, (apple.x, apple.y))
        #snake.move(random.randint(0,4))
        if(learner.mySnake.x[0] == apple.x and learner.mySnake.y[0] == apple.y):
            rew = APPLE_REWARD
            apple.relocate()
            game_board.update(learner.mySnake.get_coords(), np.array([(apple.x/44, apple.y/44)]).astype(int))
            prev_board = game_board.get_board()
            learner.mySnake.grow()
        if show: pygame.display.update()
        rews += rew
        prev_act = act
        act = learner.takeAction(obs, rew)
        game_board.update(learner.mySnake.get_coords(), np.array([(apple.x/44, apple.y/44)]).astype(int))
        prev_board = game_board.get_board()
        learner.update_replay_memory((prev_board, act, rew, game_board.get_board(), False))
        #if show: time.sleep(0.015)
        #print(learner.mySnake.x, learner.mySnake.y)
        #print("i=", i)
         
learner.model.save('model2')        
pickle.dump(learner.model.weights, f)
f.close()
pygame.quit()