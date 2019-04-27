import os

import numpy as np
import pytest

from player.GameTreeNode import GameTreeNode
from player.Heuristic import Heuristic
from player.AlphaBeta import AlphaBeta

import cProfile


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


def print_depth_1_nodes(node, best_move, nodes_generated):
    print("########## DEPTH 1 NODES ##########\n")
    print("Nodes generated: {}".format(nodes_generated))
    print("Best move = {}\n".format(best_move))
    for c in node.children:
        print("Global state generated from move {} played on board {}\n".format(c.get_board_num(), c.parent))
        print(c.state)
        print("\nMove performed on board {}:".format(c.parent))
        print("alpha value = {}\n".format(c.alpha))
        print_board(c.state[c.parent])

        if c.children:
            print("Next board: {}".format(c.get_board_num()))
            print_board(c.board)
        else:
            print("WIN STATE\n\n")


@pytest.fixture(scope='function')
def initial_board_state():
    n = GameTreeNode(INITIAL_BOARD, 5)
    n.reset_generated_nodes()
    return n


@pytest.fixture
def initial_state_generated_nodes_depth2():
    return np.load(os.path.join(NPY_OUTPUT, 'initial_state_depth2.npy'))


def test_negamax_on_filled_board_win_state_at_depth2():
    pass


# not a great test since it relies on proper ordering nodes which is done at run time.
# def test_correct_player(initial_board_state):
#     """ Checks to see that we generating the right amount of player and opponent states in the negamax algorithm """
#     m = minimax(initial_board_state, Heuristic, 3)
#     m.run()
#     player_array = np.array(m.players)
#     print(np.count_nonzero(player_array == 1))
#     print(np.count_nonzero(player_array == -1))
#     assert np.count_nonzero(player_array == 1) == 33 and np.count_nonzero(player_array == -1) == 9


def test_win_at_depth_1():
    """ Checks to see if negamax can find move to win in one turn """
    start_node = GameTreeNode(FILLED_BOARD, 4)

    m = AlphaBeta(start_node, Heuristic, 3)
    best_move = m.run()

    print_depth_1_nodes(start_node, best_move, m.nodes_generated)

    assert best_move == 2

def test_negamax_avoid_loss_in_next_turn_1():
    """ Checks to see that negamax avoids allowing the opponent to win in the next turn """

    state = np.array([ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, -1, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # The board we must make a move on
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    start_node = GameTreeNode(state, 3)
    m = AlphaBeta(start_node, Heuristic, 3)
    best_move = m.run()

    print_depth_1_nodes(start_node, best_move, m.nodes_generated)

    assert best_move != 1


def test_negamax_avoid_loss_in_next_turn_2():
    """ Checks to see that negamax avoids allowing the opponent to win in the next turn """

    state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # the board we must make a move on
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, -1, 0, -1, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])

    start_node = GameTreeNode(state, 1)
    m = AlphaBeta(start_node, Heuristic, 3)
    best_move = m.run()

    # print_depth_1_nodes(start_node, best_move, m.nodes_generated)

    assert best_move != 7

def test_avoid_loss_in_next_turn_3():
    """ Checks to see that negamax avoids allowing the opponent to win in the next turn """


    state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, -1, -1, 0, 1, 0, -1, 0, 0],
                      [0, 0, 0, 1, -1, 0, -1, 1, 0, 0],
                      [0, 0, 1, 1, 0, -1, 0, 0, 0, -1],
                      [0, 0, -1, 0, 1, 0, 0, 0, -1, 0],     # the board we are need to make a move on
                      [0, -1, 0, 0, 0, 1, 1, -1, 0, -1],
                      [0, -1, 1, 0, 0, 0, 0, -1, 1, 0],
                      [0, 1, 0, 0, 0, -1, 1, 0, -1, 1],
                      [0, 0, 0, 0, -1, 1, 0, 0, 0, 1],
                      [0, 1, 0, -1, 1, 0, -1, 0, 0, 0]])

    start_node = GameTreeNode(state, 4)
    m = AlphaBeta(start_node, Heuristic, 3)
    best_move = m.run()

    print_depth_1_nodes(start_node, best_move, m.nodes_generated)

    assert best_move != 6


def test_avoid_losing_in_next_turn_4():
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
                      [0, 1, 0, -1, 1, 0, -1, 0, 0, 0]])

    start_node = GameTreeNode(state, 4)
    m = AlphaBeta(start_node, Heuristic, 3)
    best_move = m.run()

    print_depth_1_nodes(start_node, best_move, m.nodes_generated)

    assert best_move != 6

def test_generate_best_move_opponent_depth_2():
    state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, -1, -1, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, -1, 0, 0, 0, 0],
                      [0, 0, 0, -1, 1, -1, -1, 0, 1, 1],
                      [0, 0, 1, 0, 0, 0, 0, -1, -1, 0],     # board that we make a move on
                      [0, 1, 0, 1, 0, -1, 0, 1, -1, -1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, -1, 0, 0, -1, 1, 0, 0, 0, 0],
                      [0, 1, 0, 1, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, -1, 0, 1, 0, 1, 0, -1]])

    start_node = GameTreeNode(state, 4)
    m = AlphaBeta(start_node, Heuristic, 3)
    best_move = m.run()

    print_depth_1_nodes(start_node, best_move, m.nodes_generated)

    # first_move_node = max(start_node.children, key=lambda c: c.alpha)
    # second_move_node = max(first_move_node.children, key=lambda c: c.alpha)
    # print_depth_1_nodes(first_move_node, second_move_node.move, m.nodes_generated)

    assert best_move == 6





if __name__ == "__main__":
    # cProfile.run('test_negamax_avoid_loss_in_next_turn_2()')
    test_generate_best_move_opponent_depth_2()