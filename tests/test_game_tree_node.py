import pytest
import numpy as np
from player.GameTreeNode import GameTreeNode

INITIAL_BOARD = np.zeros((10, 10), dtype="int8")
FULL_BOARD = np.ones((10, 10), dtype="int8")
PARTIAL_BOARD = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 2, 0, 2],
                          [0, 0, 0, 0, 1, 2, 0, 0, 0, 0],
                          [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 2, 0, 1, 0, 0, 0],
                          [0, 2, 2, 1, 0, 1, 0, 0, 0, 0],
                          [0, 2, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 2, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

BOARD_WITH_MULTIPLE_WINS = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],    # rows
                                     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                     [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],    # columns
                                     [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

CURRENT_PLAYER = 1


@pytest.fixture(scope='function')
def initial_board_state_node():
    return GameTreeNode(INITIAL_BOARD, 5)


@pytest.fixture
def full_board_state_node():
    return GameTreeNode(FULL_BOARD, 1)


@pytest.fixture
def partial_board_state_node():
    return GameTreeNode(PARTIAL_BOARD, 5)


@pytest.fixture
def multiple_wins_state_node():
    return GameTreeNode(BOARD_WITH_MULTIPLE_WINS, 1)


def test_init_game_tree_node(initial_board_state_node):
    g = initial_board_state_node
    assert np.array_equal(g.board, INITIAL_BOARD[5])


# def test_generate_moves_from_initial_board_state(initial_board_state_node):
#     g = initial_board_state_node
#     result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#     g.generate_moves(CURRENT_PLAYER)
#     assert np.array_equal([i.board for i in g.children], result) and np.array_equal(g.board, INITIAL_BOARD[5])


def test_no_moves_can_be_generated(full_board_state_node):
    g = full_board_state_node
    g.generate_moves(CURRENT_PLAYER)
    assert len(g.children) == 0


# def test_partial_board_state(partial_board_state_node):
#     g = partial_board_state_node
#     g.generate_moves(CURRENT_PLAYER)
#     possible_moves = np.array([[0, 2, 2, 1, 1, 1, 0, 0, 0, 0],
#                                [0, 2, 2, 1, 0, 1, 1, 0, 0, 0],
#                                [0, 2, 2, 1, 0, 1, 0, 1, 0, 0],
#                                [0, 2, 2, 1, 0, 1, 0, 0, 1, 0],
#                                [0, 2, 2, 1, 0, 1, 0, 0, 0, 1]])
#     assert np.array_equal([i.board for i in g.children], possible_moves)


def test_terminal_node_rows(multiple_wins_state_node):
    g = multiple_wins_state_node
    assert g.is_terminal_node(g.state) is True


def test_terminal_board_true():
    board = np.array([[ 0 , 0 , 1 , 0 , 1 , 1 , -1 , 0 , 1 , -1]])
    assert GameTreeNode.is_terminal_node(board) is True


def test_terminal_board_false():
    board = np.array([[ 0 , 0 , 0 , 0 , 1 , 1 , -1 , 0 , 1 , -1]])
    assert GameTreeNode.is_terminal_node(board) is False


def test_no_generated_moves_on_terminal_node():
    board = np.array([0, 0, 1, 0, 1, 1, -1, 0, 1, -1])
