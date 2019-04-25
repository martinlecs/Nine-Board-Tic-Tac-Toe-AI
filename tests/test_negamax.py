import os

import numpy as np
import pytest

from player.GameTreeNode import GameTreeNode
from player.heuristic.Ash_Heuristic import Heuristic
from player.negamax import minimax
import warnings


INITIAL_BOARD = np.zeros((10, 10), dtype="int8")
NPY_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests', 'numpy_output')

FILLED_BOARD = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, -1, 0, 0, -1, 1, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0, 0, -1, 0],
                         [0, 0, -1, 1, 0, -1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, -1, 0, 1, -1],
                         [0, 0, -1, -1, -1, 0, 0, 1, 0, 1],
                         [0, -1, 1, 1, 0, -1, 0, 0, 0, 1],
                         [0, 1, 0, -1, 1, 0, -1, 0, 0, 0],
                         [0, 0, 0, -1, 0, 0, 1, -1, 0, 1],
                         [0, 0, 0, 0, -1, 0, -1, -1, -1, 1]
                         ])


def print_board(board):

    def replace_values_with_char(board):
        new_board = list(board)
        for i in range(len(new_board)):
            if new_board[i] == 0:
                new_board[i] = '.'
            elif new_board[i] == 1:
                new_board[i] = 'O'
            else:
                new_board[i] = 'X'
        return new_board

    board = replace_values_with_char(board)
    print(board[1], board[2], board[3])
    print(board[4], board[5], board[6])
    print(board[7], board[8], board[9], end="\n\n")
    print(" ------+-------+------")


@pytest.fixture(scope='function')
def initial_board_state():
    n = GameTreeNode(INITIAL_BOARD, 5)
    n.reset_generated_nodes()
    return n


@pytest.fixture
def initial_state_generated_nodes_depth2():
    return np.load(os.path.join(NPY_OUTPUT, 'initial_state_depth2.npy'))


def test_negamax_filled_board_win_state_at_depth_1():

    def print_children(generated, depth, header=True):

        if header:
            print("*****Depth {} Children:*****\n".format(depth))

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            depth1_children = generated
            for j in depth1_children:
                print("Heuristic value = {}\n".format(j[1]))
                print_board(j[0])


    current_board = 4
    game_node = GameTreeNode(FILLED_BOARD, current_board)

    # print("*****Current board:*****")
    # print_board(game_node.board)

    state, best_move, generated = minimax(game_node, Heuristic, 7, generated_nodes=True)

    # Depth 1
    # print_children(generated[0], 1)
    # print_children(generated[1], 2)
    # for c in game_node.children:
        # print_board(c.board)
    #
    # print("*****Selected Move:*****")
    # print("Move: {}".format(best_move))
    # print("Updated state: ")
    # print_board(state[current_board])
    assert best_move == 2


def test_negamax_on_filled_board_win_state_at_depth2():
    pass


