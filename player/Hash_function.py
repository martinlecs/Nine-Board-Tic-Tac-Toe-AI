import numpy as np


def __hash_function(board: np.ndarray):
    hash = 0
    for x in range(1, 10):
        hash +=  (board[x]*3**x)

    return hash

if __name__ == "__main__":
    h = __hash_function([0, 1, 0, 0, 1, -1, 0, 0, 0, 0])

