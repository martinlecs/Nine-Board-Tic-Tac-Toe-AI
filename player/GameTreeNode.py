import os

import numpy as np

NPY_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests', 'numpy_output')
ALL_OUTPUT = []


class GameTreeNode:
    """GameTreeNode is used to generate successor states for the negamax search algorithm.

    Attributes:
        state (np.ndarray): a 10x10 numpy array representing the world state of Tic-Tac-Toe
        board (int): A number used to represent the current board that we are playing on

    """

    def __init__(self, state: np.ndarray, board: int, move=None):
        self._state = state
        self._board = board
        self._children = None
        self._move = move

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
        return self._move

    def update_state_with_move(self, move, player):
        self._state[self._board][move] = player

    def get_size_children(self):
        return len(self._children)

    def order_moves(self):
        # need to use heuristic on heuristic
        # possibly order moves based on which board has best position
        pass

    @staticmethod
    def all_generated_nodes(save=False):
        generated = np.array(ALL_OUTPUT)
        if save:
            np.save(os.path.join(NPY_OUTPUT, 'initial_state_depth2'), generated)
        return generated

    @staticmethod
    def reset_generated_nodes():
        global ALL_OUTPUT
        ALL_OUTPUT = []

    def is_terminal_node(self):
        """ Check is a state is a win-state for the player """

        def check_equal(lst):
            lst = list(lst)
            no_zeroes = True if lst[0] != 0 else False
            return no_zeroes and lst.count(lst[0]) == len(lst)

        board = self.board
        # check rows for win state
        rows = any([check_equal(board[1:4]), check_equal(board[4:7]), check_equal(board[7:10])])
        # check columns for win state
        columns = any([check_equal(board[[1, 4, 7]]), check_equal(board[[2, 5, 8]]), check_equal(board[[3, 6, 9]])])
        # check diagonals for win state
        diagonals = any([check_equal(board[[1, 5, 9]]), check_equal(board[[3, 5, 7]])])
        return any([rows, columns, diagonals])

    def generate_moves(self, player):
        """ Generates all possible moves for current player by looking at empty squares as potential moves
            Player 1 = 1, Player 2 = -1

        """

        def create_new_successor_node(state, board_num, move, player):
            state_copy = np.copy(state)
            state_copy[self._board][move] = player
            return GameTreeNode(state_copy, board_num, move=move)

        board = self.board
        move_list = []
        for i in range(1, len(board)):
            if board[i] == 0:
                move_list.append(create_new_successor_node(self._state, self._board, i, player))
                continue
        self._children = move_list
        ALL_OUTPUT.append(np.array([i.board for i in self._children]))
