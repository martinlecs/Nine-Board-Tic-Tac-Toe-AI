import os

import numpy as np
import pytest

from player.GameTreeNode import GameTreeNode
from player.Heuristic import Heuristic
from player.negamax import minimax

INITIAL_BOARD = np.zeros((10, 10), dtype="int8")
NPY_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests', 'numpy_output')


@pytest.fixture(scope='function')
def initial_board_state():
    n = GameTreeNode(INITIAL_BOARD, 5)
    n.reset_generated_nodes()
    return n


@pytest.fixture
def initial_state_generated_nodes_depth2():
    return np.load(os.path.join(NPY_OUTPUT, 'initial_state_depth2.npy'))


@pytest.fixture(scope='function')
def board_one_move_to_win_node():
    node = GameTreeNode(np.array([0, 1, 0, -1, 0, 1, -1, 1, -1, 0]))
    node.reset_generated_nodes()
    return node


# TODO: fix once heuristic has been implemented
# def test_best_move_from_initial_board_state(initial_board_state):
#     best_move = minimax(initial_board_state, Heuristic.heuristic, 1)
#     assert best_move[1] == 1


def test_generated_moves_from_initial_board_state_depth2(initial_board_state, initial_state_generated_nodes_depth2):
    best_move = minimax(initial_board_state, Heuristic.heuristic, 2, generated_nodes=True)
    assert all([np.array_equal(initial_state_generated_nodes_depth2[i], best_move[2][i]) for i in
                range(len(initial_state_generated_nodes_depth2))])

# def test_generate_move_from_partially_full_board(board_one_move_to_win_node):
#     best_move = minimax(board_one_move_to_win_node, Heuristic.heuristic, 2, generated_nodes=True)
#     assert best_move[1] == 4 and best_move[2][0].shape == (3, 10) and best_move[2][1].shape == (2, 10)
