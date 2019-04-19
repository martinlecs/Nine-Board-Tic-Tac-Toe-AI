import numpy as np


#TODO: is there an advantage to making this class over the global np array?

class GameState:

    def __init__(self, state: np.ndarray):
        self._state = state
        self._number_of_moves = 0

    def update_board(self):
        pass
