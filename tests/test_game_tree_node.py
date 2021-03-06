import pytest
import numpy as np
from player.Game import Game
from player.GameTreeNode import GameTreeNode
from player.Heuristic import Heuristic

CURRENT_PLAYER = 1
INITIAL_BOARD = np.zeros((10, 10), dtype="i1")
FULL_BOARD = np.ones((10, 10), dtype="i1")
PARTIAL_BOARD = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, -1, 0, -1],
                          [0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
                          [0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, -1, 0, 1, 0, 0, 0],
                          [0, -1, -1, 1, 0, 1, 0, 0, 0, 0],
                          [0, -1, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype='i1')

BOARD_WITH_MULTIPLE_WINS = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],    # rows
                                     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                     [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],    # columns
                                     [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='i1')


@pytest.fixture
def heuristic_func():
    h = Heuristic()
    h.load()
    return h


@pytest.fixture(scope='function')
def game_cls():
    g = Game()
    g.load()
    return g


@pytest.fixture(scope='function')
def initial_board_state_node(game_cls):
    parameterized_board = np.array([game_cls.board_to_hash(b) for b in INITIAL_BOARD])
    return GameTreeNode(parameterized_board, 5)


@pytest.fixture
def full_board_state_node():
    return GameTreeNode(FULL_BOARD, 1)


@pytest.fixture
def partial_board_state_node():
    return GameTreeNode(PARTIAL_BOARD, 5)


@pytest.fixture
def multiple_wins_state_node():
    return GameTreeNode(BOARD_WITH_MULTIPLE_WINS, 1)


# def test_init_game_tree_node(initial_board_state_node):
#     g = initial_board_state_node
#     assert np.array_equal(g.board, INITIAL_BOARD[5])
#
#
# def test_no_moves_can_be_generated(full_board_state_node, heuristic_func, game_cls):
#     g = full_board_state_node
#     g.generate_moves(CURRENT_PLAYER, heuristic_func, 1)
#     assert len(g.children) == 0
#
#
# def test_terminal_node_rows(game_cls, multiple_wins_state_node):
#     g = multiple_wins_state_node
#     assert game_cls.is_terminal(g.state) is True
#
#
# def test_terminal_board_true(game_cls):
#     board = np.array([[ 0 , 0 , 1 , 0 , 1 , 1 , -1 , 0 , 1 , -1]], dtype='i1')
#     assert game_cls.is_terminal(board) is True
#
#
# def test_terminal_board_false(game_cls):
#     board = np.array([[ 0 , 0 , 0 , 0 , 1 , 1 , -1 , 0 , 1 , -1]], dtype='i1')
#     assert game_cls.is_terminal(board) is False



