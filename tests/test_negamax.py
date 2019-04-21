import numpy as np
import pytest
from player.GameTreeNode import GameTreeNode
from player.negamax import minimax
from player.Heuristic import Heuristic
import os

INITIAL_BOARD = np.zeros((10, 10), dtype="int8")
NPY_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests', 'numpy_output')


@pytest.fixture(scope='function')
def initial_board_state():
    n = GameTreeNode(INITIAL_BOARD[5])
    n.reset_generated_nodes()
    return n

@pytest.fixture
def initial_state_generated_nodes_depth2():
    return np.load(os.path.join(NPY_OUTPUT, 'initial_state_depth2.npy'))


def test_best_move_from_initial_board_state(initial_board_state):
    best_move = minimax(initial_board_state, Heuristic.heuristic, 1)
    assert best_move[1] == 1


def test_generated_moves_from_initial_board_state_depth2(initial_board_state, initial_state_generated_nodes_depth2):
    best_move = minimax(initial_board_state, Heuristic.heuristic, 2, generated_nodes=True)
    assert all([np.array_equal(initial_state_generated_nodes_depth2[i], best_move[2][i]) for i in range(len(initial_state_generated_nodes_depth2))])
