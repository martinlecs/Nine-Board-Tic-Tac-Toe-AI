import numpy as np


def __hash_function(board: np.ndarray):
    """ Calculates the heuristic value for each vertical in a Tic-Tac-Toe board.


    Args:
        board (numpy.ndarray): The current Tic-Tac-Toe board that we are currently playing on,
                              np.ndarray has shape (10,).

    Returns:
        vertical_one, vertical_two, opponent_vertical_one, opponent_vertical_two, winner, loser
        (int, int, int, int, int, int): Tuple contains the heuristic values calculated for a Tic-Tac-Toe board's verticals.

    """
    hash = 0
    for x in range(1, 10):
        hash +=  (board[x]*3**x)

    return hash

if __name__ == "__main__":
    h = __hash_function([0, 1, 0, 0, 1, -1, 0, 0, 0, 0])

