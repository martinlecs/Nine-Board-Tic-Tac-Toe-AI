import numpy as np
import os

NPY_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests', 'numpy_output')
ALL_OUTPUT = []


class GameTreeNode:

    def __init__(self, board: np.ndarray, parent=None, move=None):
        self._board = board
        self._parent = parent
        self._children = None
        self._move = move


    def is_terminal_node(self):
        """ Check is a state is a win-state for the player """

        def check_equal(lst):
            lst = list(lst)
            no_zeroes = True if lst[0] != 0 else False
            return no_zeroes and lst.count(lst[0]) == len(lst)

        board = self._board
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

        board = self._board
        move_list = []
        for i in range(1, len(board)):
            if board[i] == 0:
                new_move = np.copy(board)
                new_move[i] = player
                # print(new_move)
                move_list.append(GameTreeNode(new_move, move=i))
                continue
        # print("\n")
        self._children = move_list
        ALL_OUTPUT.append(np.array([i.get_board() for i in self._children]))

    def order_moves(self):
        # need to use heuristic on heuristic
        # possibly order moves based on which board has best position
        pass

    def get_children(self):
        return self._children

    def get_size_children(self):
        return len(self._children)

    def get_board(self):
        return self._board

    def set_parent(self, parent):
        self._parent = parent

    def get_move(self):
        return self._move

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

