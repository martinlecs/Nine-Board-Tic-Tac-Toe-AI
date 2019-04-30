import numpy as np


PARTIAL_BOARD = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, -1, 0, 1, 0, 0, 1],
                          [0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
                          [0, 1, 0, 0, 1, -1, 0, 0, 0, 0],
                          [0, 0, 0, -1, 0, -1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


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

        Method:
            (For this explanation, we are assuming that we are X and the opponent is O)
            The number of diagonals that have 2 Xs are counted, and the number of diagonals with 1 x
            and no Os are counted. Similarly the number of diagonals that have 2 Os and no Xs
            are counted and the number of diagonals that have 1 Os and no Xs are also counted.
            And lastly, the diagonal where we win (ie 3 Xs) and the diagonal where we lose
            (ie 3 Os) are also counted.
            This is done by the consideration of every possible combination in the 2 diagonals (ie the number of ways
            I can have 2 Xs and no Os,the number of ways I can have 1 X and no O and the number of ways I can win
            with 3 Xs, and the same is considered for the opponent as well). And under each circumstance,
            the appropriate variable gets incremented (ie if a diagonal with 2 Xs and no Os is found, diagonal_two
            is incremented and if a diagonal with 2 Os and no Xs is found, then opponent_diagonal_two is incremented and etc).
        """

        # TODO: replace this ugly variable initialisation with a dict
        diagonal_one = diagonal_two = opponent_diagonal_one = opponent_diagonal_two = winner = loser = 0

        if board[1] == 1 and board[5] == 0 and board[9] == 0:
            ##print ("In diagonal if1")
            diagonal_one += 1
        elif board[1] == 0 and board[5] == 1 and board[9] == 0:
            ##print ("In diagonal if2")
            diagonal_one += 1
        elif board[1] == 0 and board[5] == 0 and board[9] == 1:
            ##print ("In diagonal if3")
            diagonal_one += 1
        elif board[1] == 1 and board[5] == 1 and board[9] == 0:
            ##print ("In diagonal if4")
            diagonal_two += 1
        elif board[1] == 0 and board[5] == 1 and board[9] == 1:
            ##print ("In diagonal if5")
            diagonal_two += 1
        elif board[1] == 1 and board[5] == 0 and board[9] == 1:
            ##print ("In diagonal if6")
            diagonal_two += 1
        elif board[1] == 1 and board[5] == 1 and board[9] == 1:
            ##print ("In diagonal if7")
            winner += 1
        if board[3] == 1 and board[5] == 0 and board[7] == 0:
            ##print ("In diagonal if8")
            diagonal_one += 1
        elif board[3] == 0 and board[5] == 1 and board[7] == 0:
            ##print ("In diagonal if9")
            diagonal_one += 1
        elif board[3] == 0 and board[5] == 0 and board[7] == 1:
            ##print ("In diagonal if10")
            diagonal_one += 1
        elif board[3] == 1 and board[5] == 1 and board[7] == 0:
            ##print ("In diagonal if11")
            diagonal_two += 1
        elif board[3] == 0 and board[5] == 1 and board[7] == 1:
            ##print ("In diagonal if12")
            diagonal_two += 1
        elif board[3] == 1 and board[5] == 0 and board[7] == 1:
            ##print ("In diagonal if13")
            diagonal_two += 1
        elif board[3] == 1 and board[5] == 1 and board[7] == 1:
            ##print ("In diagonal if14")
            winner += 1

        if board[1] == -1 and board[5] == 0 and board[9] == 0:
            ##print ("In diagonal if15")
            opponent_diagonal_one += 1
        elif board[1] == 0 and board[5] == -1 and board[9] == 0:
            ##print ("In diagonal if16")
            opponent_diagonal_one += 1
        elif board[1] == 0 and board[5] == 0 and board[9] == -1:
            ##print ("In diagonal if17")
            opponent_diagonal_one += 1
        elif board[1] == -1 and board[5] == -1 and board[9] == 0:
            ##print ("In diagonal if18")
            opponent_diagonal_two += 1
        elif board[1] == 0 and board[5] == -1 and board[9] == -1:
            ##print ("In diagonal if19")
            opponent_diagonal_two += 1
        elif board[1] == -1 and board[5] == 0 and board[9] == -1:
            ##print ("In diagonal if20")
            opponent_diagonal_two += 1
        elif board[1] == -1 and board[5] == -1 and board[9] == -1:
            ##print ("In diagonal if21")
            loser += 1
        if board[3] == -1 and board[5] == 0 and board[7] == 0:
            ##print ("In diagonal if22")
            opponent_diagonal_one += 1
        elif board[3] == 0 and board[5] == -1 and board[7] == 0:
            ##print ("In diagonal if23")
            opponent_diagonal_one += 1
        elif board[3] == 0 and board[5] == 0 and board[7] == -1:
            ##print ("In diagonal if24")
            opponent_diagonal_one += 1
        elif board[3] == -1 and board[5] == -1 and board[7] == 0:
            ##print ("In diagonal if25")
            opponent_diagonal_two += 1
        elif board[3] == 0 and board[5] == -1 and board[7] == -1:
            ##print ("In diagonal if26")
            opponent_diagonal_two += 1
        elif board[3] == -1 and board[5] == 0 and board[7] == -1:
            ##print ("In diagonal if27")
            opponent_diagonal_two += 1
        elif board[3] == -1 and board[5] == -1 and board[7] == -1:
            ##print ("In diagonal if28")
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
        Method:

            (For this explanation, we are assuming that we are X and the opponent is O)
            The number of columns that have 2 Xs are counted, and the number of columns with 1 x
            and no Os are counted. Similarly the number of columns that have 2 Os and no Xs
            are counted and the number of columns that have 1 Os and no Xs are also counted.
            And lastly,the column where we win (ie 3 Xs) and the column where we lose
            (ie 3 Os) are also counted.
            This is done by the consideration of every possible combination in the 2 columns (ie the number of ways
            I can have 2 Xs and no Os,the number of ways I can have 1 X and no O and the number of ways I can win
            with 3 Xs, and the same is considered for the opponent as well). And under each circumstance,
            the appropriate variable gets incremented (ie if a column with 2 Xs and no Os is found, vertical_two
            is incremented and if a column with 2 Os and no Xs is found, then opponent_vertical_two is incremented and etc).
        """

        vertical_one = vertical_two = opponent_vertical_one = opponent_vertical_two = winner = loser = 0

        for x in range(1, 4):
            if board[x] == 1 and board[x + 3] == 0 and board[x + 6] == 0:
                #print ("in vertical if 1 with column %d" %(x))
                vertical_one += 1
            elif board[x] == 0 and board[x + 3] == 1 and board[x + 6] == 0:
                #print("in vertical if 2 with column %d" % (x))
                vertical_one += 1
            elif board[x] == 0 and board[x + 3] == 0 and board[x + 6] == 1:
                #print("in vertical if 3 with column %d" % (x))
                vertical_one += 1
            elif board[x] == 1 and board[x + 3] == 1 and board[x + 6] == 0:
                #print("in vertical if 4 with column %d" % (x))
                vertical_two += 1
            elif board[x] == 0 and board[x + 3] == 1 and board[x + 6] == 1:
                #print("in vertical if 5 with column %d" % (x))
                vertical_two += 1
            elif board[x] == 1 and board[x + 3] == 0 and board[x + 6] == 1:
                #print("in vertical if 6 with column %d" % (x))
                vertical_two += 1
            elif board[x] == 1 and board[x + 3] == 1 and board[x + 6] == 1:
                #print("in vertical if 7 with column %d" % (x))
                winner += 1

            if board[x] == -1 and board[x + 3] == 0 and board[x + 6] == 0:
                #print("in vertical if 8 with column %d" % (x))
                opponent_vertical_one += 1
            elif board[x] == 0 and board[x + 3] == -1 and board[x + 6] == 0:
                #print("in vertical if 9 with column %d" % (x))
                opponent_vertical_one += 1
            elif board[x] == 0 and board[x + 3] == 0 and board[x + 6] == -1:
                #print("in vertical if 10 with column %d" % (x))
                opponent_vertical_one += 1
            elif board[x] == -1 and board[x + 3] == -1 and board[x + 6] == 0:
                #print("in vertical if 11 with column %d" % (x))
                opponent_vertical_two += 1
            elif board[x] == 0 and board[x + 3] == -1 and board[x + 6] == -1:
                #print("in vertical if 12 with column %d" % (x))
                opponent_vertical_two += 1
            elif board[x] == -1 and board[x + 3] == 0 and board[x + 6] == -1:
                #print("in vertical if 13 with column %d" % (x))
                opponent_vertical_two += 1
            elif board[x] == -1 and board[x + 3] == -1 and board[x + 6] == -1:
                #print("in vertical if 14 with column %d" % (x))
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

        Method:
            (For this explanation, we are assuming that we are X and the opponent is O)
            The number of rows that have 2 Xs are counted, and the number of rows with 1 x
            and no Os are counted. Similarly the number of rows that have 2 Os and no Xs
            are counted and the number of rows that have 1 Os and no Xs are also counted.
            And lastly,the row where we win (ie 3 Xs) and the row where we lose
            (ie 3 Os) are also counted.
            This is done by the consideration of every possible combination in the 2 rows (ie the number of ways
            I can have 2 Xs and no Os,the number of ways I can have 1 X and no O and the number of ways I can win
            with 3 Xs, and the same is considered for the opponent as well). And under each circumstance,
            the appropriate variable gets incremented (ie if a row with 2 Xs and no Os is found, horizontal_two
            is incremented and if a row with 2 Os and no Xs is found, then opponent_horizontal_two is incremented and etc).
        """

        horizontal_one = horizontal_two = opponent_horizontal_one = opponent_horizontal_two = winner = loser = 0

        digits = [1, 4, 7]
        for x in digits:
            if board[x] == 1 and board[x + 1] == 0 and board[x + 2] == 0:
                #print("in horizontal if 1 with row %d" % (x))
                horizontal_one += 1
            elif board[x] == 0 and board[x + 1] == 1 and board[x + 2] == 0:
                #print("in horizontal if 2 with row %d" % (x))
                horizontal_one += 1
            elif board[x] == 0 and board[x + 1] == 0 and board[x + 2] == 1:
                #print("in horizontal if 3 with row %d" % (x))
                horizontal_one += 1
            elif board[x] == 1 and board[x + 1] == 1 and board[x + 2] == 0:
                #print("in horizontal if 4 with row %d" % (x))
                horizontal_two += 1
            elif board[x] == 0 and board[x + 1] == 1 and board[x + 2] == 1:
                #print("in horizontal if 5 with row %d" % (x))
                horizontal_two += 1
            elif board[x] == 1 and board[x + 1] == 0 and board[x + 2] == 1:
                #print("in horizontal if 6 with row %d" % (x))
                horizontal_two += 1
            elif board[x] == 1 and board[x + 1] == 1 and board[x + 2] == 1:
                #print("in horizontal if 7 with row %d" % (x))
                winner += 1

            if board[x] == -1 and board[x + 1] == 0 and board[x + 2] == 0:
                #print("in horizontal if 8 with row %d" % (x))
                opponent_horizontal_one += 1
            elif board[x] == 0 and board[x + 1] == -1 and board[x + 2] == 0:
                #print("in horizontal if 9 with row %d" % (x))
                opponent_horizontal_one += 1
            elif board[x] == 0 and board[x + 1] == 0 and board[x + 2] == -1:
                #print("in horizontal if 10 with row %d" % (x))
                opponent_horizontal_one += 1
            elif board[x] == -1 and board[x + 1] == -1 and board[x + 2] == 0:
                #print("in horizontal if 11 with row %d" % (x))
                opponent_horizontal_two += 1
            elif board[x] == 0 and board[x + 1] == -1 and board[x + 2] == -1:
                #print("in horizontal if 12 with row %d" % (x))
                opponent_horizontal_two += 1
            elif board[x] == -1 and board[x + 1] == 0 and board[x + 2] == -1:
                #print("in horizontal if 13 with row %d" % (x))
                opponent_horizontal_two += 1
            elif board[x] == -1 and board[x + 1] == -1 and board[x + 2] == -1:
                #print("in horizontal if 14 with row %d" % (x))
                loser += 1

        return horizontal_one, horizontal_two, opponent_horizontal_one, opponent_horizontal_two, winner, loser

    def __calculate_board_heuristic(self, board):
        """ Calculates the board heuristic for a single board.

        Args:
            board (numpy.ndarray): Numpy Representation of a single Tic-Tac-Toe board with shape (10,)

        Returns:
            heuristic (int): Heuristic value of the board

        Method:

            Here we calculating the total heuristic for the particular sub-board. Here we first try to find out
            the number of rows, columns and diagonals in the sub-board that have 2 Xs and no Os (and this value
            is stored in my_two), the number of rows, columns and diagonals that have 1 X and no Os (and this
            value is stored in my_one) and the number of rows, columns and diagonals in the sub-board where we
            have won (and this value is stored in winner). Subsequently, we also calculate these values for the
            opponent, where we calculate the number of rows, columns and diagonals in the sub-board that have 2 Os
            and no Xs (and the value is stored in opp_two), the number of rows, columns and diagonals that have
            1 O and no Xs (and the value is stored in opp_one) and the number of rows, columns and diagonals, we lose
            (and the value is stored in loser).
            We compute these values by using the functions above (calculate_vertical, calculate_horizontal and
            calculate_diagonal). After computing these values, we multiply these values by the subsequent alpha, beta,
            gamma, delta, win and lose- which are stored as global variables above. We multiply in the form of
            heuristic = win*winner + lose*loser + alpha*my_two + beta*my_one - gamma*opp_two - delta*opp_one.
            After calculating the heuristic for the sub-board, the heuristic is returned.

        """
        diagonal_one, diagonal_two, opponent_diagonal_one, opponent_diagonal_two, winner_d, loser_d = self.__calculate_diagonal(board)
        vertical_one, vertical_two, opponent_vertical_one, opponent_vertical_two, winner_v, loser_v = self.__calculate_vertical(board)
        horizontal_one, horizontal_two, opponent_horizontal_one, opponent_horizontal_two, winner_h, loser_h = self.__calculate_horizontal(board)

        winner = winner_d + winner_v + winner_h
        loser = loser_d + loser_v + loser_h
        my_two = vertical_two + diagonal_two + horizontal_two
        my_one = vertical_one  + diagonal_one + horizontal_one
        opp_two = opponent_vertical_two + opponent_diagonal_two + opponent_horizontal_two
        opp_one = opponent_vertical_one + opponent_diagonal_one + opponent_horizontal_one

        heuristic = self._win * winner + self._lose * loser + self._alpha * my_two + self._beta * my_one - self._gamma * opp_two - self._delta * opp_one

        #print(my_two)
        #print (my_one)
        #print(opp_two)
        #print (opp_one)
        #print ("****")
        return heuristic

    def heuristic(self):
        """ Calculates the total heuristic value for the global board.

        Returns:
            heuristic (int): Heuristic value of the global board

        Method:
            Given a global state, this computes the heuristic for every single sub-board (using the function calculate_board_heuristic)
            and adds them all up to give the global heuristic (ie the heuristic for the entire state).

        """
        total_heuristic = 0
        for board in self._global_board:
            #print (total_heuristic)
            #print("!!!!!")
            total_heuristic += self.__calculate_board_heuristic(board)

        return total_heuristic


if __name__ == "__main__":
    h = Heuristic(PARTIAL_BOARD)
    print_board(PARTIAL_BOARD)
    print(h.heuristic())
