import numpy as np
import itertools
import pickle
import os
from typing import Tuple

SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Heuristic:
    """ Calculates the heuristic value for the global board. Used in Alpha Beta Search search.

     Description:
        Alpha, beta, gamma, delta, win and lose are the values we use to calculate the heuristic of the global
        board. The heuristic for a sub board is calculated according the formula:

            heuristic = win*winner + lose*loser + alpha*my_two + beta*my_one - gamma*opp_two - delta*opp_one.

        Given the heuristic equation, the reason why the values are chosen in such a way, to help the alpha-
        beta algorithm.

        In this heuristic, winning in the next move is preferred the most, hence if a board with
        the winning state, the overpowering part of the heuristic of that board, will be the winning value of the
        heuristic (win*winner==> and since winner = 1, win*winner will not equal to 0, hence it will be the most
        overpoewring value in the equation).

        The next overpowering value will be the lose value of the heuristic equation (ie. if the global board has a
        losing state, then lose*loser will be greater than 0- since loser will not equal to 0) and hence granted that
        the global state doesnt not have a winning state, the losing state will be the most overpowering value in the
        board's heuristic.

        The reason as to why lose is negative, is because we want that value to be negative, so that the alpha-beta
        wont pick the move that results in that value (since we are max). Similarly, granted that the board doesnt have
        a winning or a losing state, the next overpowering value will be gamma*opp_two, in the heuristic
        (granted that there is are one or more rows, columns and/or diagonals that have 2 Os and no Xs).

        This is because we dont want the agent to be stuck in a position, where the opponent has guaranteed victory,
        because no matter where we move, in the next board, the opponent will win. Hence we try to avoid the global
        state where the opponent has managed to have rows, columns and/or diagonals with 2 Os and no Xs are in
        multiple boards. Again, just like before, gamma*opp_two is subtracted from the heuristic, to ensure that that
        value will effect the total heuristic, negatively, causing the algorithm to not prefer it.

        The next state the we prefer, is the state, where the agent has managed to create a global board where
        multiple sub-boards have rows, columns and/or diagonals with 2Xs and no Os. This is why the next highest value
        or overpowering value in the heuristic is alpha*my_two (granted that there are no winning or losing states in
        the global state, and there are very none to very few rows, columns and/or diagonals with 2 Os and no Xs).

        This value (ie alpha*my_two) is added to the overall heuristic value, to ensure that it has a positive influence
        to the overall heuristic value of the global board. And lastly,we have beta and delta, which are multiplied with
        my_one (ie. the value that stores the number of rows, columns and/or diagonals that have 1 X and no Os) and
        opp_one (ie. the value that stores the number of rows, columns and/or diagonals that have 1 O and no Xs),
        respectively.

        Since opp_two is in regards to the opponent, it is subtracted from the heuristic, to have a negative
        effect on the overall heuristic value for the board, and similarly,since my_two is to do with the agent, it is
        added to the overall heuristic, to ensure that it could have a positive effect on the overall heuristic value of
        the global board. Therefore the coefficients from most influential (negative or positive) on the overall
        heuristic value of the global value, to the least influential are win (positive influence),
        lose (negative influence), gamma (negative influence), alpha (positive influence),
        beta (positive influence) and delta (negative influence) (beta and delta should have the same level of
        influence on the overall heuristic value).

    Attributes:
        _precalc_boards (dict[int: int]): Maps board hash to a precalculated heuristic value.

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

    def load(self):
        """ Loads hashed heuristic files used during search to minimise computation """

        if not self._precalc_boards:
            try:
                # load heuristic_values dict mapping hash values -> heuristic values
                with open(os.path.join(SAVE_PATH, 'heuristic_values.pickle'), 'rb') as file:
                    self._precalc_boards = pickle.load(file)

            except FileNotFoundError as e:
                self.__precompute_heuristic_values()

    def set_params(self, alpha: int, beta: int, gamma: int, delta: int, win: int, lose: int):
        """ Sets parameters for heuristic function """

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._win = win
        self._lose = lose

    @staticmethod
    def calculate_diagonal(board: np.ndarray) -> Tuple[int, int, int, int, int, int]:
        """ Calculates the heuristic value for each diagonal in a Tic-Tac-Toe board.

        (For this explanation, we are assuming that we are X and the opponent is O)

        The number of diagonals that have 2 Xs are counted, and the number of diagonals with 1 x and no Os are counted.
        Similarly the number of diagonals that have 2 Os and no Xs are counted and the number of diagonals that have
        1 Os and no Xs are also counted. And lastly, the diagonal where we win (ie 3 Xs) and the diagonal where we lose
        (ie 3 Os) are also counted.

        This is done by the consideration of every possible combination in the 2 diagonals (ie the number of ways
        I can have 2 Xs and no Os,the number of ways I can have 1 X and no O and the number of ways I can win
        with 3 Xs, and the same is considered for the opponent as well). And under each circumstance,
        the appropriate variable gets incremented (ie if a diagonal with 2 Xs and no Os is found,
        diagonal_two is incremented and if a diagonal with 2 Os and no Xs is found, then opponent_diagonal_two is
        incremented and etc).

        Arguments:
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

    @staticmethod
    def __calculate_vertical(board: np.ndarray) -> Tuple[int, int, int, int, int, int]:
        """ Calculates the heuristic value for each vertical in a Tic-Tac-Toe board.

        (For this explanation, we are assuming that we are X and the opponent is O)

        The number of columns that have 2 Xs are counted, and the number of columns with 1 x and no Os are counted.
        Similarly the number of columns that have 2 Os and no Xs are counted and the number of columns that have 1 Os
        and no Xs are also counted. And lastly,the column where we win (ie 3 Xs) and the column where we lose
        (ie 3 Os) are also counted. This is done by the consideration of every possible combination in the 2 columns
        (ie the number of ways I can have 2 Xs and no Os,the number of ways I can have 1 X and no O and the number of
        ways I can win with 3 Xs, and the same is considered for the opponent as well).

        Under each circumstance, the appropriate variable gets incremented (ie if a column with 2 Xs and no Os is found,
        vertical_two is incremented and if a column with 2 Os and no Xs is found, then opponent_vertical_two is
        incremented and etc).

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
    def __calculate_horizontal(board: np.ndarray) -> Tuple[int, int, int, int, int, int]:
        """ Calculates the heuristic value for each horizontal in a Tic-Tac-Toe board.

        (For this explanation, we are assuming that we are X and the opponent is O)

        The number of rows that have 2 Xs are counted, and the number of rows with 1 x and no Os are counted.
        Similarly the number of rows that have 2 Os and no Xs are counted and the number of rows that have 1 Os and
        no Xs are also counted.

        Lastly,the row where we win (ie 3 Xs) and the row where we lose (ie 3 Os) are also counted.

        This is done by the consideration of every possible combination in the 2 rows (ie the number of ways
        I can have 2 Xs and no Os,the number of ways I can have 1 X and no O and the number of ways I can win
        with 3 Xs, and the same is considered for the opponent as well). And under each circumstance,
        the appropriate variable gets incremented (ie if a row with 2 Xs and no Os is found, horizontal_two
        is incremented and if a row with 2 Os and no Xs is found, then opponent_horizontal_two is incremented and etc).

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

    def __calculate_board_heuristic(self, board: np.ndarray) -> int:
        """ Calculates the board heuristic for a single board.

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

        Args:
            board (numpy.ndarray): Numpy Representation of a single Tic-Tac-Toe board with shape (10,)

        Returns:
            heuristic (int): Heuristic value of the board

        """
        diagonal_one, diagonal_two, opponent_diagonal_one, opponent_diagonal_two, winner_d, loser_d = self.calculate_diagonal(
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

    def compute_heuristic(self, global_board: np.ndarray, depth: int) -> float:
        """ Calculates the total heuristic value for the global board.

        Given a global state, this computes the heuristic for every single sub-board
        (using the function calculate_board_heuristic) and adds them all up to give the global heuristic
        (ie the heuristic for the entire state).

        Arguments:
            global_board (numpy array): Numpy representation of the 10x10 global state.
            depth (int): Depth that heuristic was calculated at.

        Returns:
            heuristic (float): Heuristic value of the global board

        """
        return sum([self._precalc_boards[b] for b in global_board]) / depth

    def __precompute_heuristic_values(self):
        """ Precomputes heuristic values and saves them into a file for later access. """

        # generate all possible boards
        result = itertools.product([0, 1, -1], repeat=9)  # creates a generator

        heuristic_dict = {}
        for counter, i in enumerate(result):
            i = list(i)
            i.insert(0, 0)  # add leading zero to match formatting of np.array in agent.py
            if i.count(1) < 5 or i.count(-1) < 5:
                heuristic_dict[counter] = self.__calculate_board_heuristic(np.array(i, dtype='i1'))

        with open(os.path.join(SAVE_PATH, 'heuristic_values.pickle'), 'wb') as file:
            pickle.dump(heuristic_dict, file)

        self._precalc_boards = heuristic_dict


