import pytest
import numpy as np
from player.GameTreeNode import GameTreeNode

INITIAL_BOARD = np.zeros((10, 10), dtype="int8")


@pytest.fixture
def initial_board_state_node():
    return GameTreeNode(INITIAL_BOARD, 5)

@pytest.fixture
def full_board_state_node():
    return GameTreeNode(INITIAL_BOARD, 1)


def test_init_game_tree_node(initial_board_state_node):
    g = initial_board_state_node
    assert np.array_equal(g.get_state(), INITIAL_BOARD)


def test_generate_moves_from_initial_board_state(initial_board_state_node):
    g = initial_board_state_node
    result = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    g.generate_moves()
    assert np.array_equal(g.get_children(), result) and np.array_equal(g.get_state(), INITIAL_BOARD)

def test_no_moves_can_be_generated():
    pass
