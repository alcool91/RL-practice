# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:03:45 2020

@author: Web
"""
p1_wins = 0; p2_wins = 0; draw_count = 0
# system libs
import argparse
import multiprocessing as mp
import tkinter as tk

# 3rd party libs
import numpy as np

# Local libs
from Player import AIPlayer, RandomPlayer, HumanPlayer

#https://stackoverflow.com/a/37737985
def turn_worker(board, send_end, p_func):
    send_end.send(p_func(board))


class Game:
    def __init__(self, player1, player2, time, games):
        self.players = [player1, player2]
        self.colors = ['yellow', 'red']
        self.current_turn = 0
        self.board = np.zeros([6,7]).astype(np.uint8)
        self.gui_board = []
        self.game_over = False
        self.end_of_game = False
        self.ai_turn_limit = time
        self.games = 0
        self.draw = False
        self.n = 0
        self.max_games = games

        #https://stackoverflow.com/a/38159672
        root = tk.Tk()
        root.title('Connect 4')
        self.root = root
        self.player_string = tk.Label(root, text=player1.player_string)
        self.player_string.pack()
        self.c = tk.Canvas(root, width=700, height=600)
        self.c.pack()

        for row in range(0, 700, 100):
            column = []
            for col in range(0, 700, 100):
                column.append(self.c.create_oval(row, col, row+100, col+100, fill=''))
            self.gui_board.append(column)

        tk.Button(root, text='Next Move', command=self.play_series_of_games).pack()

        root.mainloop()
        
    def play_series_of_games(self):
        while self.games < self.max_games:
            self.play_whole_game()
        
    def play_whole_game(self):
        while not self.end_of_game:
            self.make_move()
        self.end_of_game = False
    def make_move(self):
        if not self.game_over:
            current_player = self.players[self.current_turn]

            if current_player.type == 'ai':
                
                if self.players[int(not self.current_turn)].type == 'random':
                    p_func = current_player.get_expectimax_move
                    #p_func = current_player.get_alpha_beta_move
                else:
                    p_func = current_player.get_alpha_beta_move
                
                try:
                    recv_end, send_end = mp.Pipe(False)
                    p = mp.Process(target=turn_worker, args=(self.board, send_end, p_func))
                    p.start()
                    if p.join(self.ai_turn_limit) is None and p.is_alive():
                        p.terminate()
                        raise Exception('Player Exceeded time limit')
                except Exception as e:
                    uh_oh = 'Uh oh.... something is wrong with Player {}'
                    print(uh_oh.format(current_player.player_number))
                    print(e)
                    raise Exception('Game Over')

                move = recv_end.recv()
                # print(self.n)
                # self.n += 1
            else:
                move = current_player.get_move(self.board)

            if move is not None:
                self.update_board(int(move), current_player.player_number)
                print("board update called!")

            if self.game_completed(current_player.player_number):
                self.game_over = True
                self.player_string.configure(text=self.players[self.current_turn].player_string + ' wins!')
                print(self.players[self.current_turn].player_string + ' wins!')
                if self.draw:
                    self.draw = False
                    print("It's a draw!")
                elif self.current_turn == 0:
                    global p1_wins; global p2_wins
                    print(self.players[self.current_turn].player_string + ' wins!')
                    p1_wins += 1
                else:
                    print(self.players[self.current_turn].player_string + ' wins!')
                    p2_wins += 1
                self.board = np.zeros([6,7])
                self.games += 1
                self.reset_board()
                print("board reset and game counter incremented!")
                if self.games < self.max_games:
                    self.game_over = False
                    self.end_of_game = True
                else:
                    print("p1: " , p1_wins)
                    print("p2: " , p2_wins)
                    print("draws: ", draw_count)
                
            else:
                self.current_turn = int(not self.current_turn)
                self.player_string.configure(text=self.players[self.current_turn].player_string)

    def update_board(self, move, player_num):
        if 0 in self.board[:,move]:
            update_row = -1
            for row in range(1, self.board.shape[0]):
                update_row = -1
                if self.board[row, move] > 0 and self.board[row-1, move] == 0:
                    update_row = row-1
                elif row==self.board.shape[0]-1 and self.board[row, move] == 0:
                    update_row = row

                if update_row >= 0:
                    self.board[update_row, move] = player_num
                    self.c.itemconfig(self.gui_board[move][update_row],
                                      fill=self.colors[self.current_turn])
                    self.root.update()
                    print("board updated -- ")
                    break
        else:
            err = 'Invalid move by player {}. Column {}'.format(player_num, move)
            raise Exception(err)
            
    def reset_board(self):
        for i in range(len(self.gui_board)):
            for j in range(len(self.gui_board[0])):
                self.c.itemconfig(self.gui_board[i][j], fill='')


    def game_completed(self, player_num):
        global draw_count
        player_win_str = '{0}{0}{0}{0}'.format(player_num)
        board = self.board
        to_str = lambda a: ''.join(a.astype(str))
        
        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b
                
                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1]-3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        if not 0 in self.board and not ((check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))):
            self.draw = True
            draw_count += 1
            return True

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))



def main(player1, player2, time, games):
    """
    Creates player objects based on the string paramters that are passed
    to it and calls play_game()

    INPUTS:
    player1 - a string ['ai', 'random', 'human']
    player2 - a string ['ai', 'random', 'human']
    """
    def make_player(name, num):
        if name=='ai':
            return AIPlayer(num)
        elif name=='random':
            return RandomPlayer(num)
        elif name=='human':
            return HumanPlayer(num)

    Game(make_player(player1, 1), make_player(player2, 2), time, games)


def play_game(player1, player2):
    """
    Creates a new game GUI and plays a game using the two players passed in.

    INPUTS:
    - player1 an object of type AIPlayer, RandomPlayer, or HumanPlayer
    - player2 an object of type AIPlayer, RandomPlayer, or HumanPlayer

    RETURNS:
    None
    """
    board = np.zeros([6,7])



if __name__=='__main__':
    player_types = ['ai', 'random', 'human']
    parser = argparse.ArgumentParser()
    parser.add_argument('player1', choices=player_types)
    parser.add_argument('player2', choices=player_types)
    parser.add_argument('--time',
                        type=int,
                        default=60,
                        help='Time to wait for a move in seconds (int)')
    args = parser.parse_args()
    
    main(args.player1, args.player2, args.time, 100)
    print(p1_wins)