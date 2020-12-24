# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:05:00 2020

@author: Allen
"""

import numpy as np
import sys
import math
debug = False

class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.opponent_number = 1 if self.player_number == 2 else 2
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)
        max_score = -99999999
        max_j = []
        for j in valid_cols:
            imaginary_board = board.copy()
            for i in range(len(imaginary_board)):
                if imaginary_board[i][j] != 0:
                    imaginary_board[i-1][j] = self.player_number
                    temp_score = self.minimax(imaginary_board, 4, -math.inf, math.inf, False)
                    print(j, temp_score)
                    if temp_score >= max_score:
                        if temp_score != max_score:
                            max_j = []
                        max_score = temp_score
                        max_j.append(j)
                    break
                if i == len(imaginary_board)-1:
                    imaginary_board[i][j] = self.player_number
                    temp_score = self.minimax(imaginary_board, 4, -math.inf, math.inf, False)
                    print(j, temp_score)
                    if temp_score >= max_score:
                        if temp_score != max_score:
                            max_j = []
                        max_score = temp_score
                        max_j.append(j)
                    
                
        # print("choosing move expectimax")
        # print(self.evaluation_function(board))
        # sys.stdout.flush()
        try: 
            return np.random.choice(max_j)
        except:
            print(board)
        #raise NotImplementedError('Whoops I don\'t know what to do')

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)
        max_score = -99999999
        max_j = []
        for j in valid_cols:
            imaginary_board = board.copy()
            for i in range(len(imaginary_board)):
                if imaginary_board[i][j] != 0:
                    imaginary_board[i-1][j] = self.player_number
                    temp_score = self.minimax(imaginary_board, 4, -math.inf, math.inf, False)
                    print(j, temp_score)
                    if temp_score >= max_score:
                        if temp_score != max_score:
                            max_j = []
                        max_score = temp_score
                        max_j.append(j)
                    break
                if i == len(imaginary_board)-1:
                    imaginary_board[i][j] = self.player_number
                    temp_score = self.minimax(imaginary_board, 4, -math.inf, math.inf, False)
                    print(j, temp_score)
                    if temp_score >= max_score:
                        if temp_score != max_score:
                            max_j = []
                        max_score = temp_score
                        max_j.append(j)
                    
                
        # print("choosing move expectimax")
        # print(self.evaluation_function(board))
        # sys.stdout.flush()
        try: 
            return np.random.choice(max_j)
        except:
            print(board)

    def sim_move(self, board, j, player):
        for i in range(len(board)):
            if board[i][j] != 0:
                board[i-1][j] = player
                break
            if i == len(board)-1:
                board[i][j] = player   

    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        score, terminal = self.evaluation_function(board)
        if depth == 0 or terminal == True:
            return score
        if maximizingPlayer:
            value = -math.inf
            valid_cols = []
            for col in range(board.shape[1]):
                if 0 in board[:,col]:
                    valid_cols.append(col)
            for j in valid_cols:
                imaginary_board = board.copy()
                self.sim_move(imaginary_board, j, self.player_number)
                value = np.max([value, self.minimax(imaginary_board, depth-1, alpha, beta, False)])
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            valid_cols = []
            for col in range(board.shape[1]):
                if 0 in board[:,col]:
                    valid_cols.append(col)
            for j in valid_cols:
                imaginary_board = board.copy()
                self.sim_move(imaginary_board, j, self.opponent_number)
                value = np.min([value, self.minimax(imaginary_board, depth-1, alpha, beta, True)])
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value
            
    def expectimax(self, board, depth, maximizingPlayer):
        score, terminal = self.evaluation_function(board)
        if depth == 0 or terminal == True:
            return score
        if maximizingPlayer:
            value = -math.inf
            valid_cols = []
            for col in range(board.shape[1]):
                if 0 in board[:,col]:
                    valid_cols.append(col)
            for j in valid_cols:
                imaginary_board = board.copy()
                self.sim_move(imaginary_board, j, self.player_number)
                value = np.max([value, self.minimax(imaginary_board, depth-1, False)])
            return value
        else:
            value = math.inf
            valid_cols = []
            for col in range(board.shape[1]):
                if 0 in board[:,col]:
                    valid_cols.append(col)
            #p = 1/len(valid_cols)
            for j in valid_cols:
                imaginary_board = board.copy()
                self.sim_move(imaginary_board, j, self.opponent_number)
                value = np.mean([value, self.minimax(imaginary_board, depth-1, True)])
            return value


    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        whether the current board represents a terminal condition
        """
        score = 0
        terminal = False
        rows = len(board)
        cols = len(board[0])
        #print(board)
        ##############################################
        #Score Rows
        #1000 for 4 in a row, 100 for 3 in a row, 10 for 2 in a row (as long as it can be completed)
        #
        for i in range(rows):
            for j in range(cols-3):
                unused_count = 0
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
                    if debug: print(i, j, "4-p,h")
                    score += 1000 
                    terminal = True
                elif opponent_count == 4:
                    if debug: print(i, j, "4-o,h")
                    score -= 2000 
                    terminal = True
                elif player_count == 3 and unused_count == 1:
                    if debug: print(i, j, "3-p,h")
                    score += 100 
                elif opponent_count == 3 and unused_count == 1:
                    if debug: print(i, j, "3-o,h")
                    score -= 200 
                elif player_count == 2 and unused_count == 2:
                    if debug: print(i, j, "2-p,h")
                    score += 10 
                elif opponent_count == 2 and unused_count == 2:
                    if debug: print(i, j, "2-o,h")
                    score -= 20
                 
        ##############################################
        #Score Columns
        #1000 for 4 in a row, 100 for 3 in a row, 10 for 2 in a row (as long as it can be completed)
        #
        for j in range(cols):
            for i in range(rows-3):
                unused_count = 0
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
                    if debug: print(i, j, "4-p,v")
                    score += 10000 
                    terminal = True
                elif opponent_count == 4:
                    if debug: print(i, j, "4-o,v")
                    score -= 20000 
                    terminal = True
                elif player_count == 3 and unused_count == 1:
                    if debug: print(i, j, "3-p,v")
                    score += 100 
                elif opponent_count == 3 and unused_count == 1:
                    if debug: print(i, j, "3-o,v")
                    score -= 200 
                elif player_count == 2 and unused_count == 2:
                    if debug: print(i, j, "2-p,v")
                    score += 10 
                elif opponent_count == 2 and unused_count == 2:
                    if debug: print(i, j, "2-o,v")
                    score -= 20
                    
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
                    score += 10000 
                    terminal = True
                elif opponent_count == 4:
                    score -= 20000 
                    terminal = True
                elif player_count == 3 and unused_count == 1:
                    score += 100 
                elif opponent_count == 3 and unused_count == 1:
                    score -= 200 
                elif player_count == 2 and unused_count == 2:
                    score += 10 
                elif opponent_count == 2 and unused_count == 2:
                    score -= 20
                    
                    
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
                    score += 10000 
                    terminal = True
                elif opponent_count == 4:
                    score -= 20000 
                    terminal = True
                elif player_count == 3 and unused_count == 1:
                    score += 100 
                elif opponent_count == 3 and unused_count == 1:
                    score -= 200 
                elif player_count == 2 and unused_count == 2:
                    score += 10 
                elif opponent_count == 2 and unused_count == 2:
                    score -= 20
        if not np.any(board[0] == 0):
            terminal = True
        if debug: print(score, terminal)
        return score, terminal


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move
