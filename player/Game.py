import itertools
import os
import pickle
import numpy as np

SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ClassNotLoaded(Exception):
    pass


class Game:
    """ Contains all game-related accessory methods """

    def __init__(self):
        self._win_states = None
        self._board_hashes = None

    def load(self):
        """ Loads necessary precomputed values into class for easy access """

        if not self._win_states:
            try:
                # load board dict mapping numpy board states -> internal hash value
                with open(os.path.join(SAVE_PATH, 'board_hashes.pickle'), 'rb') as file:
                    self._board_hashes = pickle.load(file)

                # load heuristic_values dict mapping hash values -> heuristic values
                with open(os.path.join(SAVE_PATH, 'win_states.pickle'), 'rb') as file:
                    self._win_states = pickle.load(file)

            except FileNotFoundError:
                self.__precompute_board_hash()
                self.__precompute_win_states()

    def board_hash_value(self, board: np.ndarray):
        """ Returns the corresponding hash value for a board """

        return self._board_hashes[np.array(board).astype('i1').tostring()]

    def is_terminal(self, state: np.ndarray):
        """ Checks if there is a terminal node in a given global game state """

        return any([self._win_states[self._board_hashes[s.tostring()]] for s in state])

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

    def __precompute_board_hash(self):
        """ Function generates a dictionary containing all possible board variations and assigns each board a hash
            The hash value is used to represent the board during the search and for calculating the heuristic without
            having the need to store the entire board.

        """

        # generate all possible states
        num_to_select = 9  # number of squares in tic-tac-toe board
        possible_values = [0, 1, -1]
        result = itertools.product(possible_values, repeat=num_to_select)  # creates a generator

        board_hash = {}
        for i in result:
            i = list(i)
            i.insert(0, 0)  # add leading zero to match formatting of np.array in agent.py
            board_hash[np.array(i, dtype='i1').tostring()] = hash(np.array(i).astype('i1').tostring())

        with open(os.path.join(SAVE_PATH, 'board_hashes.pickle'), 'wb') as file:
            pickle.dump(board_hash, file)

        self._board_hashes = board_hash

    def __precompute_win_states(self):
        """ Precomputes all possible win states and stores them into a hash for faster computation. """

        if not self._board_hashes:
            raise FileNotFoundError("board_hashes.pickle was not generated or found.")

        win_states_dict = {}

        # must use a hash function that only acts on values
        for board, hash_value in self._board_hashes.items():
            win_states_dict[hash_value] = Game.is_terminal_node(np.frombuffer(board, dtype='i1'))

        with open(os.path.join(SAVE_PATH, 'win_states.pickle'), 'wb') as file:
            pickle.dump(win_states_dict, file)

        self._win_states = win_states_dict


if __name__ == "__main__":
    g = Game()
    g.load()
