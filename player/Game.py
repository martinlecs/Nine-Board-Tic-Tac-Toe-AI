import itertools
import os
import pickle
import numpy as np
from player.Heuristic import Heuristic
from player.GameTreeNode import GameTreeNode

SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ClassNotLoaded(Exception):
    pass


class Game:
    """ Contains all game-related accessory methods """

    def __init__(self):
        self._win_states = None
        self._board_hashes = None
        self._hash_to_board = None

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

            self._hash_to_board = {v: k for k, v in self._board_hashes.items()}

    def board_to_hash(self, board: np.ndarray):
        """ Returns the corresponding hash value for a board
            board (numpy array): i1
        """

        return self._board_hashes[board.tostring()]

    def is_terminal(self, state: np.ndarray):
        """ Checks if there is a terminal node in a given global game state """
        return any([self._win_states[s] for s in state])

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
        for counter, i in enumerate(result):
            i = list(i)
            i.insert(0, 0)  # add leading zero to match formatting of np.array in agent.py
            if i.count(1) < 5 or i.count(-1) < 5:
                board_hash[np.array(i, dtype='i1').tostring()] = counter

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

    def generate_moves(self, state: np.ndarray, curr_board: int, player: int, eval_fn: Heuristic, depth: int ):
        """ Generates all possible moves for current player by looking at empty squares as potential moves
            Player 1 = 1, Player 2 = -1

        """

        # create local copy current board in play
        board = np.frombuffer(self._hash_to_board[state[curr_board]], dtype='i1')   # read-only
        modifiable_board = np.empty_like(board)
        modifiable_board[:] = board

        move_list = []
        for i in range(1, 10):
            if modifiable_board[i] == 0:

                # set board
                modifiable_board[i] = player

                # create a copy of global state and pass that down to the child
                updated_state = np.empty_like(state)
                updated_state[:] = state
                updated_state[curr_board] = self.board_to_hash(modifiable_board)

                # calculate child's heuristic value and append child to parent
                g = GameTreeNode(updated_state, i, player)
                depth = 1 if depth == 0 else depth
                g.heuristic_val = eval_fn.compute_heuristic(updated_state, depth)
                move_list.append(g)

                # reset board
                modifiable_board[i] = 0

        # order children
        reversed = True if player == 1 else False
        move_list.sort(key=lambda x: x.heuristic_val, reverse=reversed)

        return move_list


if __name__ == "__main__":
    g = Game()
    g.load()

    h = Heuristic()
    h.load()

    g.generate_moves(np.array([0, 137, 6561, 1467, 0, 4376, 6561, 14661, 18, 738]), 3, 1, h, 8)
