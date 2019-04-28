#!/usr/bin/python3
# Adapted from Sample starter bot by Zac Partrdige
# 06/04/19

import socket
import sys
import time
import numpy as np
from player.AlphaBeta import AlphaBeta
from player.Heuristic import Heuristic
from player.Game import Game
from player.GameTreeNode import GameTreeNode


class Agent:

    def __init__(self, game: Game, heuristic: Heuristic):
        self._game = game
        self._heuristic = heuristic
        self._boards = np.zeros((10, 10), dtype="int8")
        self._curr = 0
        self._player = 1

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

        n = AlphaBeta(GameTreeNode(self._boards, self._curr), self._game, self._heuristic, 3).run()
        self.place(self._curr, n, self._player)
        return n

    def place(self, board, num, player):
        """ Place a move in the global boards"""
        self._curr = num
        self._boards[board][num] = player
        # self.print_board(self._boards)

    def parse(self, string):
        """ Reads what the server has sent us and only parses the strings that are necessary """
        if "(" in string:
            command, args = string.split("(")
            args = args.split(")")[0]
            args = args.split(",")
        else:
            command, args = string, []

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
            # print("Yay!! We win!! :)")
            return -1
        elif command == "loss":
            # print("We lost :(")
            return -2
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
                if response == -1 or response == -2:
                    s.close()
                    return response
                elif response > 0:
                    s.sendall((str(response) + "\n").encode())

if __name__ == "__main__":
    HEURISTIC = Heuristic()
    GAME = Game()
    HEURISTIC.load()
    GAME.load()

    a = Agent(GAME, HEURISTIC)
    a.run()
