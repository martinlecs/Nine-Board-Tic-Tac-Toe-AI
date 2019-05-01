import os
import pickle

import numpy as np
import pytest

from player.AlphaBeta import AlphaBeta
from player.Game import Game
from player.GameTreeNode import GameTreeNode
from player.Heuristic import Heuristic


# FILE PATHS
NPY_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests', 'numpy_output')
SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load in a Python dictionary that maps hash values of boards to their actual Numpy representation.
with open(os.path.join(SAVE_PATH, 'hash_board.pickle'), 'rb') as file:
    hash_to_board = pickle.load(file)

# Preconfigured boards
INITIAL_BOARD = np.zeros((10, 10), dtype="i1")
FILLED_BOARD = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, -1, 0, 0, -1, 1, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0, 0, -1, 0],
                         [0, 0, -1, 1, 0, -1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, -1, 0, 1, -1],
                         [0, 0, -1, -1, -1, 0, 0, 1, 0, 1],
                         [0, -1, 1, 1, 0, -1, 0, 0, 0, 1],
                         [0, 1, 0, -1, 1, 0, -1, 0, 0, 0],
                         [0, 0, 0, -1, 0, 0, 1, -1, 0, 1],
                         [0, 0, 0, 0, -1, 0, -1, -1, -1, 1]], dtype='i1')


# Accessory Functions for debugging purposes

def print_board_row(board: np.ndarray, a: int, b: int, c: int, i: int, j: int, k: int):
    """ Print board row """

    chars = {0: '.', 1: 'O', -1: 'X'}

    print("", chars[board[a][i]], chars[board[a][j]], chars[board[a][k]], end=" | ")
    print(chars[board[b][i]], chars[board[b][j]], chars[board[b][k]], end=" | ")
    print(chars[board[c][i]], chars[board[c][j]], chars[board[c][k]])


def print_global_board(board: np.ndarray):
    """ Print an entire board """
    print_board_row(board, 1, 2, 3, 1, 2, 3)
    print_board_row(board, 1, 2, 3, 4, 5, 6)
    print_board_row(board, 1, 2, 3, 7, 8, 9)
    print(" ------+-------+------")
    print_board_row(board, 4, 5, 6, 1, 2, 3)
    print_board_row(board, 4, 5, 6, 4, 5, 6)
    print_board_row(board, 4, 5, 6, 7, 8, 9)
    print(" ------+-------+------")
    print_board_row(board, 7, 8, 9, 1, 2, 3)
    print_board_row(board, 7, 8, 9, 4, 5, 6)
    print_board_row(board, 7, 8, 9, 7, 8, 9)
    print()


def print_board(board: np.ndarray):
    """ Accessory function used to print board statre """

    new_board = list(board)
    for i in range(len(new_board)):
        if new_board[i] == 0:
            new_board[i] = '.'
        elif new_board[i] == 1:
            new_board[i] = 'O'
        else:
            new_board[i] = 'X'

    print(new_board[1], new_board[2], new_board[3])
    print(new_board[4], new_board[5], new_board[6])
    print(new_board[7], new_board[8], new_board[9], end="\n\n")
    print(" ------+-------+------")


def print_depth_1_nodes(node: GameTreeNode, best_move: int, nodes_generated: int):
    """ Accessory function used to print detailed information regarding all the depth 1 nodes generated in the
        alpha beta search.

    Args:
        node (GameTreeNode): The GameTreeNode we passed into the alpha beta search as the root.
        best_move (int): The best move that was chosen by the alpha beta search.
        nodes_generated (int): The total number of nodes generated by the alpha beta search.
    """
    print("########## DEPTH 1 NODES ##########\n")
    print("Nodes generated: {}".format(nodes_generated))
    print("Best move = {}\n".format(best_move))
    for c in node.children:
        print("Global state generated from move {} played on board {}\n".format(c.get_board_num(), c.parent))
        print_global_board(np.array([hash_to_board[i] for i in c.state]))
        print("\nMove performed on board {}:".format(c.parent))
        print("alpha value = {}\n".format(c.alpha))
        print_board(hash_to_board[c.state[c.parent]])

        if c.children:
            print("Next board: {}".format(c.get_board_num()))
            print_board(hash_to_board[c.board])
        else:
            print("WIN STATE\n\n")


@pytest.fixture(scope='function')
def initial_board_state():
    """ Creates an instance of GameTreeNode that contains INTITIAL_BOARD as its state. """
    n = GameTreeNode(INITIAL_BOARD, 5)
    return n


@pytest.fixture
def initial_state_generated_nodes_depth2():
    """ Loads in a file that contains all nodes generated at depth2 for a specific board """
    return np.load(os.path.join(NPY_OUTPUT, 'initial_state_depth2.npy'))


@pytest.fixture(scope='function')
def heuristic_func():
    """ Instantiates a new Heuristic object and runs it load() method. """
    h = Heuristic()
    h.load()
    return h


@pytest.fixture(scope='function')
def game_cls():
    g = Game()
    g.load()
    return g


@pytest.fixture(scope='function')
def filled_board_state(game_cls: Game):
    parameterized_state = np.array([game_cls.board_to_hash(b) for b in FILLED_BOARD])
    return GameTreeNode(parameterized_state, 4)


def test_win_at_depth_1(filled_board_state: np.ndarray, game_cls: Game, heuristic_func: Heuristic):
    """ Checks to see if search can find move to win in one turn """

    m = AlphaBeta(filled_board_state, game_cls, heuristic_func, 3)
    best_move = m.run()

    try:
        assert best_move == 2
    except AssertionError:
        print_depth_1_nodes(filled_board_state, best_move, m.nodes_generated)
        raise


def test_negamax_avoid_loss_in_next_turn_1(game_cls, heuristic_func):
    """ Checks to see that negamax avoids allowing the opponent to win in the next turn """

    state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, -1, -1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # The board we must make a move on
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, -1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='i1')

    parameterized_state = np.array([game_cls.board_to_hash(b) for b in state])

    start_node = GameTreeNode(parameterized_state, 3)
    m = AlphaBeta(start_node, game_cls, heuristic_func, 5)
    best_move = m.run()

    try:
        assert best_move != 1
    except AssertionError:
        print_depth_1_nodes(start_node, best_move, m.nodes_generated)
        raise


def test_negamax_avoid_loss_in_next_turn_2(game_cls, heuristic_func):
    """ Checks to see that negamax avoids allowing the opponent to win in the next turn """

    state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # the board we must make a move on
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, -1, 0, -1, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]], dtype='i1')

    parameterized_state = np.array([game_cls.board_to_hash(b) for b in state])

    start_node = GameTreeNode(parameterized_state, 1)
    m = AlphaBeta(start_node, game_cls, heuristic_func, 5)
    best_move = m.run()

    # print_depth_1_nodes(start_node, best_move, m.nodes_generated)

    assert best_move != 7


def test_avoid_loss_in_next_turn_3(game_cls, heuristic_func):
    """ Checks to see that negamax avoids allowing the opponent to win in the next turn """

    state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, -1, -1, 0, 1, 0, -1, 0, 0],
                      [0, 0, 0, 1, -1, 0, -1, 1, 0, 0],
                      [0, 0, 1, 1, 0, -1, 0, 0, 0, -1],
                      [0, 0, -1, 0, 1, 0, 0, 0, -1, 0],  # the board we are need to make a move on
                      [0, -1, 0, 0, 0, 1, 1, -1, 0, -1],
                      [0, -1, 1, 0, 0, 0, 0, -1, 1, 0],
                      [0, 1, 0, 0, 0, -1, 1, 0, -1, 1],
                      [0, 0, 0, 0, -1, 1, 0, 0, 0, 1],
                      [0, 1, 0, -1, 1, 0, -1, 0, 0, 0]], dtype='i1')

    parameterized_state = np.array([game_cls.board_to_hash(b) for b in state])

    start_node = GameTreeNode(parameterized_state, 4)
    m = AlphaBeta(start_node, game_cls, heuristic_func, 5)
    best_move = m.run()

    # print_depth_1_nodes(start_node, best_move, m.nodes_generated)

    assert best_move != 6


def test_avoid_losing_in_next_turn_4(game_cls, heuristic_func):
    """ Checks to see that negamax avoids allowing the opponent to win in the next turn """

    state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, -1, -1, 0, 1, 0, -1, 0, 0],
                      [0, 0, 0, 1, -1, 0, -1, 1, 0, 0],
                      [0, 0, 1, 1, 0, -1, 0, 0, 0, -1],
                      [0, 0, -1, 0, 1, 0, 0, 0, -1, 0],  # the board we are need to make a move on
                      [0, -1, 0, 0, 0, 1, 1, -1, 0, -1],
                      [0, -1, 1, 0, 0, 0, 0, -1, 1, 0],
                      [0, 1, 0, 0, 0, -1, 1, 0, -1, 1],
                      [0, 0, 0, 0, -1, 1, 0, 0, 0, 1],
                      [0, 1, 0, -1, 1, 0, -1, 0, 0, 0]], dtype='i1')

    parameterized_state = np.array([game_cls.board_to_hash(b) for b in state])

    start_node = GameTreeNode(parameterized_state, 4)
    m = AlphaBeta(start_node, game_cls, heuristic_func, 5)
    best_move = m.run()

    # print_depth_1_nodes(start_node, best_move, m.nodes_generated)

    assert best_move != 6


def test_generate_best_move_opponent_depth_2(game_cls, heuristic_func):
    state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, -1, -1, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, -1, 0, 0, 0, 0],
                      [0, 0, 0, -1, 1, -1, -1, 0, 1, 1],
                      [0, 0, 1, 0, 0, 0, 0, -1, -1, 0],  # board that we make a move on
                      [0, 1, 0, 1, 0, -1, 0, 1, -1, -1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, -1, 0, 0, -1, 1, 0, 0, 0, 0],
                      [0, 1, 0, 1, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, -1, 0, 1, 0, 1, 0, -1]], dtype='i1')

    parameterized_state = np.array([game_cls.board_to_hash(b) for b in state])

    start_node = GameTreeNode(parameterized_state, 4)
    m = AlphaBeta(start_node, game_cls, heuristic_func, 5)
    best_move = m.run()

    # print_depth_1_nodes(start_node, best_move, m.nodes_generated)

    assert best_move == 6


def test_avoid_loss_in_next_move_5(game_cls, heuristic_func):
    state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, -1, 0, 1, 0, 0, 0, 1, 0, -1],
                      [0, 0, 0, 0, -1, 1, 0, -1, 0, 1],
                      [0, 0, 1, -1, 0, 0, -1, 0, 0, 0],  # move made here
                      [0, 0, 0, 0, -1, 1, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, -1, -1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                      [0, 0, -1, -1, 1, 0, 0, 0, 0, 0],
                      [0, 1, -1, 0, 0, 0, 0, 0, -1, 0],
                      [0, -1, 1, 1, 0, 0, 0, 0, 0, -1]], dtype='i1')

    parameterized_state = np.array([game_cls.board_to_hash(b) for b in state])

    start_node = GameTreeNode(parameterized_state, 3)
    m = AlphaBeta(start_node, game_cls, heuristic_func, 5)
    best_move = m.run()

    # print_depth_1_nodes(start_node, best_move, m.nodes_generated)

    assert best_move == 4


if __name__ == "__main__":
    import cProfile

    heuristic = Heuristic()
    heuristic.load()
    game = Game()
    game.load()
    # cProfile.runctx('g(x, y)', {'y': heuristic, 'x': game, 'g': test_negamax_avoid_loss_in_next_turn_2}, {})
    cProfile.runctx('g(x, y)', {'y': heuristic, 'x': game, 'g': test_avoid_loss_in_next_move_5}, {})