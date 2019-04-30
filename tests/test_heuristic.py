from player.Game import Game
from player.GameTreeNode import GameTreeNode
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
                          [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype='i1')

EMPTY_BOARD = np.array([ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='i1')

ALMOST_FULL = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, -1, -1, 0, 1, 0, -1, 0, 0],
                      [0, 0, 0, 1, -1, 0, -1, 1, 0, 0],
                      [0, 0, 1, 1, 0, -1, 0, 0, 0, -1],
                      [0, 0, -1, 0, 1, 0, 0, 0, -1, 0],
                      [0, 0, 0, -1, 0, -1, 1, -1, 0, 1],
                      [0, -1, 1, 0, 0, 0, 0, -1, 1, 0],
                      [0, 1, 0, 0, 0, -1, 1, 0, -1, 1],
                      [0, 0, 0, 0, -1, 1, 0, 0, 0, 1],
                      [0, 1, 0, -1, 1, 0, -1, 0, 0, 0]], dtype='i1')


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
def partial_board():
    return GameTreeNode(PARTIAL_BOARD, 8)


@pytest.fixture(scope='function')
def almost_full_board():
    return GameTreeNode(ALMOST_FULL, 1)


# Accessory Functions

def calculate_diagonal(board: np.ndarray):
    """ Calculates the heuristic value for each diagonal in a Tic-Tac-Toe board.


    Args:
        board (numpy.ndarray): The current Tic-Tac-Toe board that we are currently playing on,
                              np.ndarray has shape (10,).

    Returns:
        diagonal_one, diagonal_two, opponent_diagonal_one, opponent_diagonal_two, winner, loser
        (int, int, int, int, int, int): Tuple contains the heuristic values calculated for a Tic-Tac-Toe board's diagonals.

    """
    diagonal_one = diagonal_two = opponent_diagonal_one = opponent_diagonal_two = winner = loser = 0

    if board[1] == 1 and board[5] == 0 and board[9] == 0:
        diagonal_one += 1
    elif board[1] == 0 and board[5] == 1 and board[9] == 0:
        diagonal_one += 1
    elif board[1] == 0 and board[5] == 0 and board[9] == 1:
        diagonal_one += 1
    elif board[1] == 1 and board[5] == 1 and board[9] == 0:
        diagonal_two += 1
    elif board[1] == 0 and board[5] == 1 and board[9] == 1:
        diagonal_two += 1
    elif board[1] == 1 and board[5] == 0 and board[9] == 1:
        diagonal_two += 1
    elif board[1] == 1 and board[5] == 1 and board[9] == 1:
        winner += 1
    if board[3] == 1 and board[5] == 0 and board[7] == 0:
        diagonal_one += 1
    elif board[3] == 0 and board[5] == 1 and board[7] == 0:
        diagonal_one += 1
    elif board[3] == 0 and board[5] == 0 and board[7] == 1:
        diagonal_one += 1
    elif board[3] == 1 and board[5] == 1 and board[7] == 0:
        diagonal_two += 1
    elif board[3] == 0 and board[5] == 1 and board[7] == 1:
        diagonal_two += 1
    elif board[3] == 1 and board[5] == 0 and board[7] == 1:
        diagonal_two += 1
    elif board[3] == 1 and board[5] == 1 and board[7] == 1:
        winner += 1

    if board[1] == -1 and board[5] == 0 and board[9] == 0:
        opponent_diagonal_one += 1
    elif board[1] == 0 and board[5] == -1 and board[9] == 0:
        opponent_diagonal_one += 1
    elif board[1] == 0 and board[5] == 0 and board[9] == -1:
        opponent_diagonal_one += 1
    elif board[1] == -1 and board[5] == -1 and board[9] == 0:
        opponent_diagonal_two += 1
    elif board[1] == 0 and board[5] == -1 and board[9] == -1:
        opponent_diagonal_two += 1
    elif board[1] == -1 and board[5] == 0 and board[9] == -1:
        opponent_diagonal_two += 1
    elif board[1] == -1 and board[5] == -1 and board[9] == -1:
        loser += 1
    if board[3] == -1 and board[5] == 0 and board[7] == 0:
        opponent_diagonal_one += 1
    elif board[3] == 0 and board[5] == -1 and board[7] == 0:
        opponent_diagonal_one += 1
    elif board[3] == 0 and board[5] == 0 and board[7] == -1:
        opponent_diagonal_one += 1
    elif board[3] == -1 and board[5] == -1 and board[7] == 0:
        opponent_diagonal_two += 1
    elif board[3] == 0 and board[5] == -1 and board[7] == -1:
        opponent_diagonal_two += 1
    elif board[3] == -1 and board[5] == 0 and board[7] == -1:
        opponent_diagonal_two += 1
    elif board[3] == -1 and board[5] == -1 and board[7] == -1:
        loser += 1

    return diagonal_one, diagonal_two, opponent_diagonal_one, opponent_diagonal_two, winner, loser


def calculate_vertical(board: np.ndarray):
    """ Calculates the heuristic value for each vertical in a Tic-Tac-Toe board.


    Args:
        board (numpy.ndarray): The current Tic-Tac-Toe board that we are currently playing on,
                              np.ndarray has shape (10,).

    Returns:
        vertical_one, vertical_two, opponent_vertical_one, opponent_vertical_two, winner, loser
        (int, int, int, int, int, int): Tuple contains the heuristic values calculated for a Tic-Tac-Toe board's verticals.

    """

    vertical_one = vertical_two = opponent_vertical_one = opponent_vertical_two = winner = loser = 0

    for x in range(1, 4):
        if board[x] == 1 and board[x + 3] == 0 and board[x + 6] == 0:
            vertical_one += 1
        elif board[x] == 0 and board[x + 3] == 1 and board[x + 6] == 0:
            vertical_one += 1
        elif board[x] == 0 and board[x + 3] == 0 and board[x + 6] == 1:
            vertical_one += 1
        elif board[x] == 1 and board[x + 3] == 1 and board[x + 6] == 0:
            vertical_two += 1
        elif board[x] == 0 and board[x + 3] == 1 and board[x + 6] == 1:
            vertical_two += 1
        elif board[x] == 1 and board[x + 3] == 0 and board[x + 6] == 1:
            vertical_two += 1
        elif board[x] == 1 and board[x + 3] == 1 and board[x + 6] == 1:
            winner += 1

        if board[x] == -1 and board[x + 3] == 0 and board[x + 6] == 0:
            opponent_vertical_one += 1
        elif board[x] == 0 and board[x + 3] == -1 and board[x + 6] == 0:
            opponent_vertical_one += 1
        elif board[x] == 0 and board[x + 3] == 0 and board[x + 6] == -1:
            opponent_vertical_one += 1
        elif board[x] == -1 and board[x + 3] == -1 and board[x + 6] == 0:
            opponent_vertical_two += 1
        elif board[x] == 0 and board[x + 3] == -1 and board[x + 6] == -1:
            opponent_vertical_two += 1
        elif board[x] == -1 and board[x + 3] == 0 and board[x + 6] == -1:
            opponent_vertical_two += 1
        elif board[x] == -1 and board[x + 3] == -1 and board[x + 6] == -1:
            loser += 1

    return vertical_one, vertical_two, opponent_vertical_one, opponent_vertical_two, winner, loser


def calculate_horizontal(board: np.ndarray):
    """ Calculates the heuristic value for each horizontal in a Tic-Tac-Toe board.


    Args:
        board (numpy.ndarray): The current Tic-Tac-Toe board that we are currently playing on,
                              np.ndarray has shape (10,).

    Returns:
        horizontal_one, horizontal_two, opponent_horizontal_one, opponent_horizontal_two, winner, loser
        (int, int, int, int, int, int): Tuple contains the heuristic values calculated for a Tic-Tac-Toe board's horizontals.

    """

    horizontal_one = horizontal_two = opponent_horizontal_one = opponent_horizontal_two = winner = loser = 0

    digits = [1, 4, 7]
    for x in digits:
        if board[x] == 1 and board[x + 1] == 0 and board[x + 2] == 0:
            horizontal_one += 1
        elif board[x] == 0 and board[x + 1] == 1 and board[x + 2] == 0:
            horizontal_one += 1
        elif board[x] == 0 and board[x + 1] == 0 and board[x + 2] == 1:
            horizontal_one += 1
        elif board[x] == 1 and board[x + 1] == 1 and board[x + 2] == 0:
            horizontal_two += 1
        elif board[x] == 0 and board[x + 1] == 1 and board[x + 2] == 1:
            horizontal_two += 1
        elif board[x] == 1 and board[x + 1] == 0 and board[x + 2] == 1:
            horizontal_two += 1
        elif board[x] == 1 and board[x + 1] == 1 and board[x + 2] == 1:
            winner += 1

        if board[x] == -1 and board[x + 1] == 0 and board[x + 2] == 0:
            opponent_horizontal_one += 1
        elif board[x] == 0 and board[x + 1] == -1 and board[x + 2] == 0:
            opponent_horizontal_one += 1
        elif board[x] == 0 and board[x + 1] == 0 and board[x + 2] == -1:
            opponent_horizontal_one += 1
        elif board[x] == -1 and board[x + 1] == -1 and board[x + 2] == 0:
            opponent_horizontal_two += 1
        elif board[x] == 0 and board[x + 1] == -1 and board[x + 2] == -1:
            opponent_horizontal_two += 1
        elif board[x] == -1 and board[x + 1] == 0 and board[x + 2] == -1:
            opponent_horizontal_two += 1
        elif board[x] == -1 and board[x + 1] == -1 and board[x + 2] == -1:
            loser += 1

    return horizontal_one, horizontal_two, opponent_horizontal_one, opponent_horizontal_two, winner, loser


def test_heuristic_on_empty_board(heuristic_func, game_cls):
    parameterized_board = [game_cls.board_to_hash(s) for s in EMPTY_BOARD]
    assert heuristic_func.compute_heuristic(parameterized_board, 1) == 0


def test_calculate_diagonal(heuristic_func, almost_full_board):
    """ Checks that refactored calculate_diagonal is equivalent to the original """
    assert calculate_diagonal(almost_full_board.state[5]) == heuristic_func.calculate_diagonal(almost_full_board.state[5])


def test_calculate_diagonal2(heuristic_func, partial_board):
    """ Checks that refactored calculate_diagonal is equivalent to the original """
    assert calculate_diagonal(partial_board.state[8]) == heuristic_func.calculate_diagonal(partial_board.state[8])


def test_calculate_diagonal3(heuristic_func, partial_board):
    """ Checks that refactored calculate_diagonal is equivalent to the original """
    assert calculate_diagonal(partial_board.state[2]) == heuristic_func.calculate_diagonal(partial_board.state[2])

