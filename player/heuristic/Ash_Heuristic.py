import numpy as np



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


def print_board_row(board, a, b, c, i, j, k):
    # The marking script doesn't seem to like this either, so just take it out to submit
    print("", board[a][i], board[a][j], board[a][k], end = " | ")
    print(board[b][i], board[b][j], board[b][k], end = " | ")
    print(board[c][i], board[c][j], board[c][k])


def print_board(board):
    print_board_row(board, 1,2,3,1,2,3)
    print_board_row(board, 1,2,3,4,5,6)
    print_board_row(board, 1,2,3,7,8,9)
    print(" ------+-------+------")
    print_board_row(board, 4,5,6,1,2,3)
    print_board_row(board, 4,5,6,4,5,6)
    print_board_row(board, 4,5,6,7,8,9)
    print(" ------+-------+------")
    print_board_row(board, 7,8,9,1,2,3)
    print_board_row(board, 7,8,9,4,5,6)
    print_board_row(board, 7,8,9,7,8,9)
    print()


class Heuristic:
    """ Calculates the heuristic value for the global board. Used in Negamax search.

    Description of how heuristic value is calculated

    Attributes:
        global_board (numpy.ndarray): Numpy Representation of the global Tic-Tac-Toe board with shape (10,10)

    """

    def __init__(self, global_board):
        self._global_board = global_board
        self._alpha = 5
        self._beta = 1
        self._gamma = 4
        self._delta = 1
        self._win = 1000000
        self._lose = -100000

    @staticmethod
    def __calculate_diagonal(board: np.ndarray):
        """ Calculates the heuristic value for each diagonal in a Tic-Tac-Toe board.


        Args:
            board (numpy.ndarray): The current Tic-Tac-Toe board that we are currently playing on,
                                  np.ndarray has shape (10,).

        Returns:
            diagonal_one, diagonal_two, opponent_diagonal_one, opponent_diagonal_two, winner, loser
            (int, int, int, int, int, int): Tuple contains the heuristic values calculated for a Tic-Tac-Toe board's diagonals.

        """

        # TODO: replace this ugly variable initialisation with a dict
        diagonal_one = diagonal_two = opponent_diagonal_one = opponent_diagonal_two = winner = loser = 0

        if board[1] == 1 and board[5] != 1 and board[9] != 1:
            diagonal_one += 1
        elif board[1] != 1 and board[5] == 1 and board[9] != 1:
            diagonal_one += 1
        elif board[1] != 1 and board[5] != 1 and board[9] == 1:
            diagonal_one += 1
        elif board[1] == 1 and board[5] == 1 and board[9] != 1:
            diagonal_two += 1
        elif board[1] != 1 and board[5] == 1 and board[9] == 1:
            diagonal_two += 1
        elif board[1] == 1 and board[5] != 1 and board[9] == 1:
            diagonal_two += 1
        elif board[1] == 1 and board[5] == 1 and board[9] == 1:
            winner += 1
        elif board[3] == 1 and board[5] != 1 and board[7] != 1:
            diagonal_one += 1
        elif board[3] != 1 and board[5] == 1 and board[7] != 1:
            diagonal_one += 1
        elif board[3] != 1 and board[5] != 1 and board[7] == 1:
            diagonal_one += 1
        elif board[3] == 1 and board[5] == 1 and board[7] != 1:
            diagonal_two += 1
        elif board[3] != 1 and board[5] == 1 and board[7] == 1:
            diagonal_two += 1
        elif board[3] == 1 and board[5] != 1 and board[7] == 1:
            diagonal_two += 1
        elif board[3] == 1 and board[5] == 1 and board[7] == 1:
            winner += 1

        if board[1] == -1 and board[5] != -1 and board[9] != -1:
            opponent_diagonal_one += 1
        elif board[1] != -1 and board[5] == -1 and board[9] != -1:
            opponent_diagonal_one += 1
        elif board[1] != -1 and board[5] != -1 and board[9] == -1:
            opponent_diagonal_one += 1
        elif board[1] == -1 and board[5] == -1 and board[9] != -1:
            opponent_diagonal_two += 1
        elif board[1] != -1 and board[5] == -1 and board[9] == -1:
            opponent_diagonal_two += 1
        elif board[1] == -1 and board[5] != -1 and board[9] == -1:
            opponent_diagonal_two += 1
        elif board[1] == -1 and board[5] == -1 and board[9] == -1:
            loser += 1
        elif board[3] == -1 and board[5] != -1 and board[7] != -1:
            opponent_diagonal_one += 1
        elif board[3] != -1 and board[5] == -1 and board[7] != -1:
            opponent_diagonal_one += 1
        elif board[3] != -1 and board[5] != -1 and board[7] == -1:
            opponent_diagonal_one += 1
        elif board[3] == -1 and board[5] == -1 and board[7] != -1:
            opponent_diagonal_two += 1
        elif board[3] != -1 and board[5] == -1 and board[7] == -1:
            opponent_diagonal_two += 1
        elif board[3] == -1 and board[5] != -1 and board[7] == -1:
            opponent_diagonal_two += 1
        elif board[3] == -1 and board[5] == -1 and board[7] == -1:
            loser += 1

        # TODO: consider replacing this with a dict
        return diagonal_one, diagonal_two, opponent_diagonal_one, opponent_diagonal_two, winner, loser

    @staticmethod
    def __calculate_vertical(board: np.ndarray):
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
            if board[x] == 1 and board[x + 3] != 1 and board[x + 6] != 1:
                vertical_one += 1
            if board[x] != 1 and board[x + 3] == 1 and board[x + 6] != 1:
                vertical_one += 1
            if board[x] != 1 and board[x + 3] != 1 and board[x + 6] == 1:
                vertical_one += 1
            if board[x] == 1 and board[x + 3] == 1 and board[x + 6] != 1:
                vertical_two += 1
            if board[x] != 1 and board[x + 3] == 1 and board[x + 6] == 1:
                vertical_two += 1
            if board[x] == 1 and board[x + 3] != 1 and board[x + 6] == 1:
                vertical_two += 1
            if board[x] == 1 and board[x + 3] == 1 and board[x + 6] == 1:
                winner += 1

            if board[x] == -1 and board[x + 3] != -1 and board[x + 6] != -1:
                opponent_vertical_one += 1
            if board[x] != -1 and board[x + 3] == -1 and board[x + 6] != -1:
                opponent_vertical_one += 1
            if board[x] != -1 and board[x + 3] != -1 and board[x + 6] == -1:
                opponent_vertical_one += 1
            if board[x] == -1 and board[x + 3] == -1 and board[x + 6] != -1:
                opponent_vertical_two += 1
            if board[x] != -1 and board[x + 3] == -1 and board[x + 6] == -1:
                opponent_vertical_two += 1
            if board[x] == -1 and board[x + 3] != -1 and board[x + 6] == -1:
                opponent_vertical_two += 1
            if board[x] == -1 and board[x + 3] == -1 and board[x + 6] == -1:
                loser += 1

            return vertical_one, vertical_two, opponent_vertical_one, opponent_vertical_two, winner, loser

    @staticmethod
    def __calculate_horizontal(board: np.ndarray):
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
            if board[x] == 1 and board[x + 1] != 1 and board[x + 2] != 1:
                horizontal_one += 1
            if board[x] != 1 and board[x + 1] == 1 and board[x + 2] != 1:
                horizontal_one += 1
            if board[x] != 1 and board[x + 1] != 1 and board[x + 2] == 1:
                horizontal_one += 1
            if board[x] == 1 and board[x + 1] == 1 and board[x + 2] != 1:
                horizontal_two += 1
            if board[x] != 1 and board[x + 1] == 1 and board[x + 2] == 1:
                horizontal_two += 1
            if board[x] == 1 and board[x + 1] != 1 and board[x + 2] == 1:
                horizontal_two += 1
            if board[x] == 1 and board[x + 1] == 1 and board[x + 2] == 1:
                winner += 1

            if board[x] == -1 and board[x + 1] != -1 and board[x + 2] != -1:
                opponent_horizontal_one += 1
            if board[x] != -1 and board[x + 1] == -1 and board[x + 2] != -1:
                opponent_horizontal_one += 1
            if board[x] != -1 and board[x + 1] != -1 and board[x + 2] == -1:
                opponent_horizontal_one += 1
            if board[x] == -1 and board[x + 1] == -1 and board[x + 2] != -1:
                opponent_horizontal_two += 1
            if board[x] != -1 and board[x + 1] == -1 and board[x + 2] == -1:
                opponent_horizontal_two += 1
            if board[x] == -1 and board[x + 1] != -1 and board[x + 2] == -1:
                opponent_horizontal_two += 1
            if board[x] == -1 and board[x + 1] == -1 and board[x + 2] == -1:
                loser += 1

            return horizontal_one, horizontal_two, opponent_horizontal_one, opponent_horizontal_two, winner, loser

    def __calculate_board_heuristic(self, board):
        """ Calculates the board heuristic for a single board.

        Args:
            board (numpy.ndarray): Numpy Representation of a single Tic-Tac-Toe board with shape (10,)

        Returns:
            heuristic (int): Heuristic value of the board

        """
        diagonal_one, diagonal_two, opponent_diagonal_one, opponent_diagonal_two, winner_d, loser_d = self.__calculate_diagonal(board)
        vertical_one, vertical_two, opponent_vertical_one, opponent_vertical_two, winner_v, loser_v = self.__calculate_vertical(board)
        horizontal_one, horizontal_two, opponent_horizontal_one, opponent_horizontal_two, winner_h, loser_h = self.__calculate_horizontal(board)

        winner = winner_d + winner_v + winner_h
        loser = loser_d + loser_v + loser_h
        my_two = vertical_two + horizontal_two + diagonal_two
        my_one = vertical_one + horizontal_one + diagonal_one
        opp_two = opponent_vertical_two + opponent_horizontal_two + opponent_diagonal_two
        opp_one = opponent_vertical_one + opponent_horizontal_one + opponent_diagonal_one

        heuristic = self._win * winner + self._lose * loser + self._alpha * my_two + self._beta * my_one - self._gamma * opp_two - self._delta * opp_one

        return heuristic

    def heuristic(self):
        """ Calculates the total heuristic value for the global board.

        Returns:
            heuristic (int): Heuristic value of the global board

        """
        total_heuristic = 0
        for board in self._global_board:
            total_heuristic += self.__calculate_board_heuristic(board)

        return total_heuristic


if __name__ == "__main__":
    h = Heuristic(PARTIAL_BOARD)
    print_board(PARTIAL_BOARD)
    print(h.heuristic())
