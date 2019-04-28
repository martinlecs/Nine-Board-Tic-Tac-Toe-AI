import math
from typing import Callable

import numpy as np


class GameTreeNode:
    """GameTreeNode is used to generate successor states for the negamax search algorithm.

    Attributes:
        state (np.ndarray): a 10x10 numpy array representing the world state of Tic-Tac-Toe
        board (int): A number used to represent the current board that we are playing on

    """

    def __init__(self, state: np.ndarray, board: int, parent=None):

        self._state = state
        self._board = board
        self._children = None

        # For debugging purposes
        self._alpha = -math.inf
        self._beta = math.inf
        self._parent = parent

    @property
    def state(self):
        return self._state

    @property
    def board(self):
        return self._state[self._board]

    @property
    def children(self):
        return self._children

    @property
    def move(self):
        return self._board

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        self._alpha = val

    @property
    def parent(self):
        return self._parent

    def get_board_num(self):
        return self._board

    def generate_moves(self, player: int, eval_fn: Callable, depth: int ):
        """ Generates all possible moves for current player by looking at empty squares as potential moves
            Player 1 = 1, Player 2 = -1

        """

        def create_new_successor_node(state, move, player):
            state_copy = np.array(state)
            state_copy[self._board][move] = player
            return GameTreeNode(state_copy, move, parent=self._board)

        board = self.board
        move_list = []
        for i in range(1, len(board)):
            if board[i] == 0:
                move_list.append(create_new_successor_node(self._state, i, player))

        # order children
        depth = 1 if depth == 0 else depth
        move_list.sort(key=lambda x: eval_fn.compute_heuristic(x.state, depth), reverse=True)

        self._children = move_list


