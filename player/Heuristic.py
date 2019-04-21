import numpy as np

PARTIAL_BOARD = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, -1, 0, -1],
                          [0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
                          [0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, -1, 0, 1, 0, 0, 0],
                          [0, -1, -1, 1, 0, 1, 0, 0, 0, 0],
                          [0, 2, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

class Heuristic:

    def __init__(self):
        pass

    @staticmethod
    def heuristic(state: np.ndarray):
        return np.random.randint(1,9)

    @staticmethod
    def heuristic2(state: np.ndarray, player: int):
        """ Counts number of our adjacent values in board """

        def count_adjacent(array):
            return np.count_nonzero(array == 1)

        def board_value(board, player):
            board[board == -player] = 0

            rows = np.sum([np.sum(board[1:4]), np.sum(board[4:7]), np.sum(board[7:10])])

            # check columns for win board
            columns = np.sum([np.sum(board[[1, 4, 7]]), np.sum(board[[2, 5, 8]]), np.sum(board[[3, 6, 9]])])
            # # check diagonals for win board
            diagonals = np.sum([np.sum(board[[1, 5, 9]]), np.sum(board[[3, 5, 7]])])
            return np.sum([rows, columns, diagonals])

        player_1_value = board_value(state, player)
        player_2_value = board_value(state, -player)
        return player_1_value - player_2_value

    @staticmethod
    def heuristic3(state: np.ndarray):
        """ TODO: DO HEURISTIC IMPLEMENTATION HERE """
        return 0


if __name__ == "__main__":
    print(Heuristic.heuristic3(PARTIAL_BOARD))