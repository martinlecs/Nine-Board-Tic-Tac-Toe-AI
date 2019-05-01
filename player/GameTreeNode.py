import math
import numpy as np
from typing import List


class GameTreeNode:
    """GameTreeNode is used to represent game states for the Alpha Beta search algorithm.

    Arguments:
        state (np.ndarray): a 1x10 numpy array representing the global state of Tic-Tac-Toe
        board (int): A number used to represent the current board that we are playing on
        parent (int, optional): The previous board where the move was made that led to the current state

    """

    def __init__(self, state: np.ndarray, board: int, parent=None):

        self._state = state
        self._board = board
        self._parent = parent

        # list of GameTreeNodes
        self._children = []

        # Default alpha value for GameTreeNode
        self._alpha = -math.inf

    @property
    def state(self) -> np.ndarray:
        """ Numpy array: represents the current global state of the game. """
        return self._state

    @property
    def board(self) -> int:
        """ Int: hash value of the current board in play """
        return self._state[self._board]

    @property
    def children(self) -> List['GameTreeNode']:
        """ list of GameTeeNodes """
        return self._children

    @children.setter
    def children(self, lst: List['GameTreeNode']):
        """ Sets children to input value.

        Argument:
            lst (list of GameTreeNodes): List containing generated children (GameTreeNodes)
            for the current state.

        """
        self._children = lst

    @property
    def move(self) -> int:
        """ Int: The current board in play. """
        return self._board

    @property
    def alpha(self) -> float:
        """ Float: Alpha value calculated for the current node. """
        return self._alpha

    @property
    def parent(self) -> int:
        """ Returns parent board of current node. """
        return self._parent

    def get_board_num(self) -> int:
        """ Get the current board that is in play. """
        return self._board

    @parent.setter
    def parent(self, val: int):
        """ Sets parent to val. """
        self._parent = val

    @alpha.setter
    def alpha(self, val: int):
        """ Sets alpha value for node.

        Arguments:
            val (float): Alpha value calculated for current node.
        """
        self._alpha = val
