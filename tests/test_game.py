import numpy as np
import pytest
from player.Game import Game
from player.GameTreeNode import GameTreeNode


@pytest.fixture(scope='function')
def game_cls():
    g = Game()
    g.load()
    return g


def test_not_win_state(game_cls: Game):
    """ Checks that an almost empty board has no terminal nodes """
    ALMOST_BOARD = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='i1')

    parameterized_board = np.array([game_cls.board_to_hash(b) for b in ALMOST_BOARD])
    node = GameTreeNode(parameterized_board, 4)
    assert game_cls.is_terminal(node) is False

