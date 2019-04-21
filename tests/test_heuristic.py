from player.Heuristic import Heuristic
import pytest
from player.GameTreeNode import GameTreeNode
import numpy as np


@pytest.fixture
def heuristic():
    return Heuristic()


@pytest.fixture(scope='function')
def board_one_move_to_win_node_first_player():
    node = GameTreeNode(np.array([0, 1, 0, -1, 0, 1, -1, 1, -1, 0]))
    node.reset_generated_nodes()
    return node


# def test_first_player_basic(board_one_move_to_win_node_first_player):
#     heuristic_value = Heuristic.heuristic(board_one_move_to_win_node_first_player.get_board(), 1)
#     assert False

