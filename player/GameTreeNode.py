import math
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
        self._children = []

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

    @children.setter
    def children(self, lst):
        self._children = lst

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

    @parent.setter
    def parent(self, val):
        self._parent = val

    def get_board_num(self):
        return self._board


