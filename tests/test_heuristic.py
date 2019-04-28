from player.Heuristic import Heuristic
import numpy as np
import pytest


PARTIAL_BOARD = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, -1, 0, -1],
                          [0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
                          [0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, -1, 0, 1, 0, 0, 0],
                          [0, -1, -1, 1, 0, 1, 0, 0, 0, 0],
                          [0, -1, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

EMPTY_BOARD = np.array([ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


@pytest.fixture
def heuristic_func():
    h = Heuristic()
    h.load()
    return h


def test_heuristic_on_empty_board(heuristic_func):
    assert heuristic_func.compute_heuristic(EMPTY_BOARD, 1) == 0


def test_heuristic_on_partial_board(heuristic_func):
    assert heuristic_func.compute_heuristic(PARTIAL_BOARD, 1) == 7



