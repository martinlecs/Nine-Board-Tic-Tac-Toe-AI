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
        self._alpha = 45
        self._beta = 10
        self._gamma = 90
        self._delta = 10
        self._win = 1000000
        self._lose = -100000

        # debugging
        self._hash_board = None

    def load(self):
        """ Loads hashed heuristic files used during search to minimise computation """

        if not self._precalc_boards:
            try:
                # load heuristic_values dict mapping hash values -> heuristic values
                with open(os.path.join(SAVE_PATH, 'heuristic_values.pickle'), 'rb') as file:
                    self._precalc_boards = pickle.load(file)

            except FileNotFoundError as e:
                self.__precompute_heuristic_values()

    def set_params(self, alpha, beta, gamma, delta, win, lose):
        """ Sets parameters for heuristic function """

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._win = win
        self._lose = lose

    @staticmethod
    def calculate_diagonal(board: np.ndarray):
        """ Calculates the heuristic value for each diagonal in a Tic-Tac-Toe board.


        Args:
            board (numpy.ndarray): The current Tic-Tac-Toe board that we are currently playing on,
                                  np.ndarray has shape (10,).

        Returns:
            diagonal_one, diagonal_two, opponent_diagonal_one, opponent_diagonal_two, winner, loser
            (int, int, int, int, int, int): Tuple contains the heuristic values calculated for a Tic-Tac-Toe board's diagonals.

        """
        possible_states_player = np.array([np.array(i) for i in itertools.product([0, 1], repeat=3)])
        possible_states_opponent = np.negative(possible_states_player)

        diagonal_one = 1 if any(np.array_equal(board[[1, 5, 9]], x) for x in possible_states_player[[1, 2, 4]]) else 0
        diagonal_one += 1 if any(np.array_equal(board[[3, 5, 7]], x) for x in possible_states_player[[1, 2, 4]]) else 0
        diagonal_two = 1 if any(np.array_equal(board[[1, 5, 9]], x) for x in possible_states_player[[3, 5, 6]]) else 0
        diagonal_two += 1 if any(np.array_equal(board[[3, 5, 7]], x) for x in possible_states_player[[3, 5, 6]]) else 0
        winner = 1 if np.array_equal(board[[1, 5, 9]], possible_states_player[7]) else 0
        winner += 1 if np.array_equal(board[[3, 5, 7]], possible_states_player[7]) else 0

        opponent_diagonal_one = 1 if any(np.array_equal(board[[1, 5, 9]], x) for x in possible_states_opponent[[1, 2, 4]]) else 0
        opponent_diagonal_one += 1 if any(np.array_equal(board[[3, 5, 7]], x) for x in possible_states_opponent[[1, 2, 4]]) else 0
        opponent_diagonal_two = 1 if any(np.array_equal(board[[1, 5, 9]], x) for x in possible_states_opponent[[3, 5, 6]]) else 0
        opponent_diagonal_two += 1 if any(np.array_equal(board[[3, 5, 7]], x) for x in possible_states_opponent[[3, 5, 6]]) else 0
        loser = 1 if np.array_equal(board[[1, 5, 9]], possible_states_opponent[7]) else 0
        loser += 1 if np.array_equal(board[[3, 5, 7]], possible_states_opponent[7]) else 0

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

    def __calculate_board_heuristic(self, board):
        """ Calculates the board heuristic for a single board.

        Args:
            board (numpy.ndarray): Numpy Representation of a single Tic-Tac-Toe board with shape (10,)

        Returns:
            heuristic (int): Heuristic value of the board

        """
        diagonal_one, diagonal_two, opponent_diagonal_one, opponent_diagonal_two, winner_d, loser_d = self.__calculate_diagonal(
            board)
        vertical_one, vertical_two, opponent_vertical_one, opponent_vertical_two, winner_v, loser_v = self.__calculate_vertical(
            board)
        horizontal_one, horizontal_two, opponent_horizontal_one, opponent_horizontal_two, winner_h, loser_h = self.__calculate_horizontal(
            board)

        winner = winner_d + winner_v + winner_h
        loser = loser_d + loser_v + loser_h
        my_two = vertical_two + diagonal_two + horizontal_two
        my_one = vertical_one + diagonal_one + horizontal_one
        opp_two = opponent_vertical_two + opponent_diagonal_two + opponent_horizontal_two
        opp_one = opponent_vertical_one + opponent_diagonal_one + opponent_horizontal_one

        heuristic = self._win * winner + self._lose * loser + self._alpha * my_two + self._beta * my_one - self._gamma * opp_two - self._delta * opp_one

        return heuristic

    def compute_heuristic(self, global_board, depth):
        """ Calculates the total heuristic value for the global board.

        Returns:
            heuristic (int): Heuristic value of the global board

        """
        return sum([self._precalc_boards[b] for b in global_board]) / depth

    def __precompute_heuristic_values(self):

        # generate all possible states
        num_to_select = 9  # number of squares in tic-tac-toe board
        possible_values = [0, 1, -1]
        result = itertools.product(possible_values, repeat=num_to_select)  # creates a generator

        heuristic_dict = {}
        hash_board = {}
        for counter, i in enumerate(result):
            i = list(i)
            i.insert(0, 0)  # add leading zero to match formatting of np.array in agent.py
            if i.count(1) < 5 or i.count(-1) < 5:
                heuristic_dict[counter] = self.__calculate_board_heuristic(np.array(i, dtype='i1'))
                hash_board[counter] = np.array(i, dtype='i1')

        with open(os.path.join(SAVE_PATH, 'heuristic_values.pickle'), 'wb') as file:
            pickle.dump(heuristic_dict, file)

        self._precalc_boards = heuristic_dict


if __name__ == "__main__":
    possible_states_player = np.array([np.array(i) for i in itertools.product([0, 1], repeat=3)])
    print(possible_states_player)
    print(possible_states_player[[1,2,4]])
