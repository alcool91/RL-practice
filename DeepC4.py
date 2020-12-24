# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:54:33 2020

@author: Allen
"""

import numpy as np


board = np.zeros(6,7)

def move(board, player, col):
    for i in range(len(board)):
        if board[i][col] != 0:
            board[i-1][col] = player
            break
        if i == len(board)-1:
            board[i][col] = player 
            
            
def is_over(board):
    terminal = False
    rows = len(board)
    cols = len(board[0])
    for i in range(rows):
            for j in range(cols-3):
                player_count = 0
                opponent_count = 0
                for k in range(4):
                    if board[i][j+k] == self.player_number:
                        player_count += 1
                    elif board[i][j+k] == 0:
                        unused_count += 1
                    else:
                        opponent_count += 1
                if player_count == 4:
                    terminal = True
                elif opponent_count == 4:
                    terminal = True
                 
        ##############################################
        #Score Columns
        #1000 for 4 in a row, 100 for 3 in a row, 10 for 2 in a row (as long as it can be completed)
        #
        for j in range(cols):
            for i in range(rows-3):
                player_count = 0
                opponent_count = 0
                for k in range(4):
                    if board[i+k][j] == self.player_number:
                        player_count += 1
                    elif board[i+k][j] == 0:
                        unused_count += 1
                    else:
                        opponent_count += 1
                if player_count == 4:
                    terminal = True
                elif opponent_count == 4:
                    terminal = True
                    
        ##############################################
        #Score Descending Diagonals \
        #1000 for 4 in a row, 100 for 3 in a row, 10 for 2 in a row (as long as it can be completed)
        #
        for i in range(rows-3):
            for j in range(cols-3):
                unused_count = 0
                player_count = 0
                opponent_count = 0
                for k in range(4):
                    if board[i+k][j+k] == self.player_number:
                        player_count += 1
                    elif board[i+k][j+k] == 0:
                        unused_count += 1
                    else:
                        opponent_count += 1
                if player_count == 4:
                    terminal = True
                elif opponent_count == 4:
                    terminal = True
                    
                    
        ##############################################
        #Score Ascending Diagonals /
        #1000 for 4 in a row, 100 for 3 in a row, 10 for 2 in a row (as long as it can be completed)
        #
        for i in range(3, rows):
            for j in range(cols-3):
                unused_count = 0
                player_count = 0
                opponent_count = 0
                for k in range(4):
                    if board[i-k][j+k] == self.player_number:
                        player_count += 1
                    elif board[i-k][j+k] == 0:
                        unused_count += 1
                    else:
                        opponent_count += 1
                if player_count == 4:
                    terminal = True
                elif opponent_count == 4:
                    terminal = True
        if not np.any(board[0] == 0):
            terminal = True
        return terminal
            
            
            
class Q_Snake:
    
    def __init__(self, epsilon, learning_rate, discount, new_q = True):
        self.mySnake = Player(352, 484)
        self.model = self.create_model()
        
        
        
        self.replay_memory = deque(maxlen=50000)
        
        self.target_update_counter = 0
        #self.q = np.random.uniform(low=0, high=1, size=(5, 40, 40, 5))
        #self.q = np.random.uniform(low=0, high=1, size=(3, 3, 3, 3, 3, 3, 21, 5))
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.EPISODES = 300001
        self.epsilon0 = epsilon
        if not new_q:
            self.model = tf.keras.models.load_model("model1")
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
        model.add(Conv2D(32, (3,3), input_shape=(20,20,1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        #model.add(Dropout(0.2))
        
        model.add(Conv2D(32, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        #model.add(Dropout(0.2))
        
        model.add(Flatten())
        model.add(Dense(64))
        
        model.add(Dense(4, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        
        return model 
    
    def update_replay_memory(self, trans):
        self.replay_memory.append(trans)
        
    def get_qs(self, state):
        #print(state/4, state.shape)
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/4)[0]
    
    def train(self, terminal_state):
        if len(self.replay_memory) < 1000:
            return
        
        minibatch = random.sample(self.replay_memory, 128) 
        
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
        self.model.fit(np.array(X)/4, np.array(y), batch_size=128, verbose=0, shuffle=False) 
        #print(np.sum(self.model.layers[0].get_weights()[0]))
        if terminal_state:
            self.target_update_counter += 1 
            
        if self.target_update_counter > 250:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    
    def agentStart(self, obs):
        if (random.random() < self.epsilon):
            action = random.randint(0,3)
        else:
            action = np.argmax(self.get_qs(obs))
            
        
        self.last_obs = obs
        self.last_action = action
        return action
        
    def takeAction(self, obs, rew):
        if (random.random() < self.epsilon):
            action = random.randint(0,3)
        else:
            action = np.argmax(self.get_qs(obs))
            #print(self.get_qs(obs), action)
            
        
        
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
        if ep < 150000:
            hl = 150000
        elif 150000 < ep < 250000:
            hl = 50000
        else:
            hl = 15000
        #self.epsilon -= (self.epsilon0/float(self.EPISODES) - min_epsilon/float(self.EPISODES))
        self.epsilon = self.epsilon * 2**(-1/float(hl))