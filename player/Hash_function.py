import numpy as np


def __hash_function(board: np.ndarray):
    hash = 0
    for x in range(1, 10):
        #print (board[x] + 1)
        #print ((board[x] + 1)*3**x)
        hash +=  ((board[x] + 1)*3**x)

    return hash

if __name__ == "__main__":
    h = __hash_function([0, 1, 0, 0, 1, -1, 0, 0, 0, 0])
    #print(h)

