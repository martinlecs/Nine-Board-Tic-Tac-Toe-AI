import numpy as np
import itertools
import pickle
import os

SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Heuristic:
    """ Calculates the heuristic value for the global board. Used in Negamax search.

    Description of how heuristic value is calculated

    Attributes:
        global_board (numpy.ndarray): Numpy Representation of the global Tic-Tac-Toe board with shape (10,10)

    """

    def __init__(self):
        self._precalc_boards = None

        # Parameters that affect heuristic. These are the default values
        self._alpha = 5
        self._beta = 1
        self._gamma = 4
        self._delta = 1
        self._win = 1000000
        self._lose = -100000

    def load(self):
        """ Loads hashed heuristic files used during search to minimise computation """

        if not self._precalc_boards:
            try:
                # load heuristic_values dict mapping hash values -> heuristic values
                with open(os.path.join(SAVE_PATH, 'heuristic_values.pickle'), 'rb') as file:
                    self._precalc_boards = pickle.load(file)
            except Exception as e:
                self._precalc_boards = self.__precompute_heuristic_values()

    def set_params(self, alpha, beta, gamma, delta, win, lose):
        """ Sets parameters for heuristic function """

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._win = win
        self._lose = lose

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

        return heuristic

    def compute_heuristic(self, global_board, depth):
        """ Calculates the total heuristic value for the global board.

        Returns:
            heuristic (int): Heuristic value of the global board

        """
        total_heuristic = 0
        for board in global_board:
            # total_heuristic += self._precalc_boards[board.astype('i1').tostring()]
            total_heuristic += self.__calculate_board_heuristic(board)
        return total_heuristic / depth

    def __precompute_heuristic_values(self):

        # generate all possible states
        num_to_select = 9   # number of squares in tic-tac-toe board
        possible_values = [0, 1, -1]
        result = list(itertools.product(possible_values, repeat=num_to_select))

        # add extra empty element to head of list
        modified_result = []
        for i in result:
            n = list(i)
            n.insert(0, 0)
            modified_result.append(n)

        np_result = np.array(modified_result)

        # create dict that maps hash values -> heuristic values to save computation time
        heuristic_dict = {}
        for i in np_result:
            heuristic_dict[i.astype('i1').tostring()] = self.__calculate_board_heuristic(i)

        with open(os.path.join(SAVE_PATH, 'heuristic_values.pickle'), 'wb') as file:
            pickle.dump(heuristic_dict, file)

        return heuristic_dict


if __name__ == "__main__":
    h = Heuristic()
    h.load()

    PARTIAL_BOARD = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    print(h.compute_heuristic(PARTIAL_BOARD, 1))

