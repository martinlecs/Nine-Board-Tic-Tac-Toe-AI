import itertools
import os
import pickle
import numpy as np


SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Game:
    """ Contains all game-related accessory methods """

    def __init__(self):
        self._win_states = None

    def load(self):
        """ Loads necessary precomputed values into class for easy access """

        if not self._win_states:
            try:
                # load heuristic_values dict mapping hash values -> heuristic values
                with open(os.path.join(SAVE_PATH, 'win_states.pickle'), 'rb') as file:
                    self._win_states = pickle.load(file)
            except Exception as e:
                self._win_states = self.__precompute_win_states()

    def is_terminal(self, state: np.ndarray):
        """ Checks if there is a terminal node in a given global game state """

        for s in state:
            if self._win_states[s.astype('i1').tostring()]:
                return True
        return False

    @staticmethod
    def is_terminal_node(board: np.ndarray):
        """ Check is a state is a win-state for the player. ONLY TO BE USED FOR GENERATING WIN_STATES file """

        def check_equal(lst):
            lst = list(lst)
            # better to replace this with np.count_nonzeroes
            return lst[0] != 0 and lst.count(lst[0]) == len(lst)

        # check rows for win state
        rows = any([check_equal(board[1:4]), check_equal(board[4:7]), check_equal(board[7:10])])
        # check columns for win state
        columns = any([check_equal(board[[1, 4, 7]]), check_equal(board[[2, 5, 8]]), check_equal(board[[3, 6, 9]])])
        # check diagonals for win state
        diagonals = any([check_equal(board[[1, 5, 9]]), check_equal(board[[3, 5, 7]])])
        return any([rows, columns, diagonals])

    def __precompute_win_states(self):
        """ Precomputes all possible win states and saves them into a pickle file for later use. """

        def hash(board: np.ndarray):
            hash = 0
            for x in range(1, 10):
                hash += (board[x] * 3 ** x)

            return hash

        # TODO: implement O(1) lookup for hashes

        # generate all possible states
        num_to_select = 9  # number of squares in tic-tac-toe board
        possible_values = [0, 1, -1]
        result = list(itertools.product(possible_values, repeat=num_to_select))

        modified_result = []
        for i in result:
            n = list(i)
            n.insert(0, 0)
            modified_result.append(n)

        np_result = np.array(modified_result)

        # create dict that maps boards -> win states to save computation time
        win_states_dict = {}

        # must use a hash function that only acts on values
        for i in np_result:
            win_states_dict[i.astype('i1').tostring()] = Game.is_terminal_node(i)

        with open(os.path.join(SAVE_PATH, 'win_states.pickle'), 'wb') as file:
            pickle.dump(win_states_dict, file)

        return win_states_dict

if __name__ == "__main__":
    g = Game()
    g.load()
