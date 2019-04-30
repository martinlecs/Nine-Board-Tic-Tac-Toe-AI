#!/usr/bin/python3
# Adapted from Sample starter bot by Zac Partrdige
# 06/04/19

import gc
import socket
import sys
from math import ceil
import numpy as np
from player.AlphaBeta import AlphaBeta
from player.Heuristic import Heuristic
from player.Game import Game
from player.GameTreeNode import GameTreeNode


class Agent:

    def __init__(self, game: Game, heuristic: Heuristic):
        """ Agent
        :param game:
        :param heuristic:
        """
        self._game = game
        self._heuristic = heuristic
        self._boards = np.zeros(shape=(10, 10), dtype='i1')
        self._curr = 0
        self._player = 1

        self._games_played = 0
        self._games_won = 0
        self._games_drawn = 0
        self._number_moves_made = 0

    def set_heuristic_params(self, alpha, beta, gamma, delta, win, lose):
        """ Sets heuristic parameters through the Heuristic class object """

        self._heuristic.set_params(alpha, beta, gamma, delta, win, lose)

    def print_board_row(self, board, a, b, c, i, j, k):
        """ Print board row """

        chars = {0: '.', 1: 'O', -1: 'X'}

        print("", chars[board[a][i]], chars[board[a][j]], chars[board[a][k]], end = " | ")
        print(chars[board[b][i]], chars[board[b][j]], chars[board[b][k]], end = " | ")
        print(chars[board[c][i]], chars[board[c][j]], chars[board[c][k]])

    def print_board(self, board):
        """ Print an entire board """
        self.print_board_row(board, 1,2,3,1,2,3)
        self.print_board_row(board, 1,2,3,4,5,6)
        self.print_board_row(board, 1,2,3,7,8,9)
        print(" ------+-------+------")
        self.print_board_row(board, 4,5,6,1,2,3)
        self.print_board_row(board, 4,5,6,4,5,6)
        self.print_board_row(board, 4,5,6,7,8,9)
        print(" ------+-------+------")
        self.print_board_row(board, 7,8,9,1,2,3)
        self.print_board_row(board, 7,8,9,4,5,6)
        self.print_board_row(board, 7,8,9,7,8,9)
        print()

    def play(self):
        """ Choose a move to play """

        self._number_moves_made += 1

        parameterized_state = np.array([self._game.board_to_hash(b) for b in self._boards])
        node = GameTreeNode(parameterized_state, self._curr)

        n = AlphaBeta(node, self._game, self._heuristic, 7).run()
        self.place(self._curr, n, self._player)

        return n

    def place(self, board, num, player):
        """ Place a move in the global boards"""
        self._curr = num
        self._boards[board][num] = player
        # self.print_board(self._boards)

    def print_game_statistics(self):
        """ Used to print game statistics such as win rate, average number of moves made etc.
            Invoked at the end of game session.
        """

        print("#####################")
        print("Games played: {}".format(self._games_played))
        print("Games won: {}/{}".format(self._games_won, self._games_played))
        if self._games_drawn > 0:
            print("Games drawn: {}/{}".format(self._games_drawn, self._games_played))
        print('Average Number of Moves Made per Game: {}'.format(ceil(self._number_moves_made/self._games_played)))
        print("Win rate: {:.2f}%".format(self._games_won/self._games_played*100))

    def reset_boards(self):
        """ Used when playing multiple games in a row to reset the board """
        self._boards = np.zeros(shape=(10, 10), dtype='i1')
        self._curr = 0
        gc.collect()

    def parse(self, string):
        """ Reads what the server has sent us and only parses the strings that are necessary """
        if "(" in string:
            command, args = string.split("(")
            args = args.split(")")[0]
            args = args.split(",")
        else:
            command, args = string, []

        if command == "start":
            self._games_played += 1
            self.reset_boards()
        if command == "second_move":
            self.place(int(args[0]), int(args[1]), -self._player)
            return self.play()
        elif command == "third_move":
            # place the move that was generated for us
            self.place(int(args[0]), int(args[1]), -self._player)
            # place their last move
            self.place(self._curr, int(args[2]), -self._player)
            return self.play()
        elif command == "next_move":
            self.place(self._curr, int(args[0]), -self._player)
            return self.play()
        elif command == "win":
            self._games_won += 1
            print("We won!")
        elif command == "loss":
            print("We lost :(")
        elif command == "draw":
            self._games_drawn += 1
            print("Draw!")
        elif command == "end.":
            return -1
        return 0

    def run(self, port=None):
        """ Connects to a specified socket and runs the game """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = port or int(sys.argv[2])    # Usage: ./agent.py -p (port)

        s.connect(('localhost', port))
        while True:
            text = s.recv(1024).decode()
            if not text:
                continue
            for line in text.split("\n"):
                response = self.parse(line)
                if response == -1:
                    s.close()
                    return
                elif response > 0:
                    s.sendall((str(response) + "\n").encode())


if __name__ == "__main__":

    # Driver code for AI

    # Intialiase Heuristic and Game classes.
    HEURISTIC = Heuristic()
    GAME = Game()

    # Load in precalculated values
    HEURISTIC.load()
    GAME.load()

    # Initialise Agent and run the AI
    a = Agent(GAME, HEURISTIC)
    a.run()
    a.print_game_statistics()
