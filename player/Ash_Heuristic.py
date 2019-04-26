import numpy as np

alpha = 5
beta = 1
gamma = 4
delta = 1
win = 1000000
lose = -100000

class Heuristic:

    def __init__(self):
        pass

    @staticmethod
    def heuristic(globalBoard):
        total_heuristic = 0
        board_heuristic = 0
        for i in range(len(globalBoard)):
            board_heuristic = calculateBoardHeuristic(globalBoard[i])
            total_heuristic += board_heuristic
        return total_heuristic

    @staticmethod
    def calculateBoardHeuristic(board):
        winner = 0
        loser = 0
        vertical_two = 0
        vertical_one = 0
        opponent_vertical_two = 0
        opponent_vertical_one = 0
        horizontal_one = 0
        horizontal_two = 0
        opponent_horizontal_one = 0
        opponent_horizontal_two = 0
        diagonal_one = 0
        diagonal_two = 0
        opponent_diagonal_one = 0
        opponent_diagonal_two = 0
        heuristic = 0

        calculateDiagonal(board, diagonal_one, diagonal_two, opponent_diagonal_one, opponent_diagonal_two, winner, loser)
        calculateVertical (board, vertical_one, vertical_two, opponent_vertical_one, opponent_vertical_two, winner, loser)
        calculateHorizontal (board, horizontal_one, horizontal_two, opponent_horizontal_one, opponent_horizontal_two, winner, loser)

        my_two = 0
        my_one = 0
        opp_two = 0
        opp_one = 0

        my_two = vertical_two + horizontal_two + diagonal_two
        my_one = vertical_one + horizontal_one + diagonal_one
        opp_two = opponent_vertical_two + opponent_horizontal_two + opponent_diagonal_two
        opp_one = opponent_vertical_one + opponent_horizontal_one + opponent_diagonal_one
        heuristic = win*(winner) + lose*(loser) + alpha*(my_two) + beta*(my_one)- gamma* (opp_two) - delta*(opp_one)
        return heuristic

    @staticmethod
    def calculateDiagonal (board, diagonal_one : int, diagonal_two: int, opponent_diagonal_one: int, opponent_diagonal_two: int, winner: int, loser: int):

        if (board[1] == 1 && board[5] != 1 && board[9] != 1):
                diagonal_one++
        elif (board[1] != 1 && board[5] == 1 && board[9] != 1):
                diagonal_one++
        elif (board[1] != 1 && board[5] != 1 && board[9] == 1):
                diagonal_one++
        elif (board[1] == 1 && board[5] == 1 && board[9] != 1):
                diagonal_two++
        elif (board[1] != 1 && board[5] == 1 && board[9] == 1):
                diagonal_two++
        elif (board[1] == 1 && board[5] != 1 && board[9] == 1):
                diagonal_two++
        elif (board[1] == 1 && board[5] == 1 && board[9] == 1):
                winner++
        elif (board[3] == 1 && board[5] != 1 && board[7] != 1):
                diagonal_one++
        elif (board[3] != 1 && board[5] == 1 && board[7] != 1):
                diagonal_one++
        elif (board[3] != 1 && board[5] != 1 && board[7] == 1):
                diagonal_one++
        elif (board[3] == 1 && board[5] == 1 && board[7] != 1):
                diagonal_two++
        elif (board[3] != 1 && board[5] == 1 && board[7] == 1):
                diagonal_two++
        elif (board[3] == 1 && board[5] != 1 && board[7] == 1):
                diagonal_two++
        elif (board[3] == 1 && board[5] == 1 && board[7] == 1):
                winner++

        if (board[1] == -1 && board[5] != -1 && board[9] != -1):
                opponent_diagonal_one++
        elif (board[1] != -1 && board[5] == -1 && board[9] != -1):
                opponent_diagonal_one++
        elif (board[1] != -1 && board[5] != -1 && board[9] == -1):
                opponent_diagonal_one++
        elif (board[1] == -1 && board[5] == -1 && board[9] != -1):
                opponent_diagonal_two++
        elif (board[1] != -1 && board[5] == -1 && board[9] == -1):
                opponent_diagonal_two++
        elif (board[1] == -1 && board[5] != -1 && board[9] == -1):
                opponent_diagonal_two++
        elif (board[1] == -1 && board[5] == -1 && board[9] == -1):
                loser++
        elif (board[3] == -1 && board[5] != -1 && board[7] != -1):
                opponent_diagonal_one++
        elif (board[3] != -1 && board[5] == -1 && board[7] != -1):
                opponent_diagonal_one++
        elif (board[3] != -1 && board[5] != -1 && board[7] == -1):
                opponent_diagonal_one++
        elif (board[3] == -1 && board[5] == -1 && board[7] != -1):
                opponent_diagonal_two++
        elif (board[3] != -1 && board[5] == -1 && board[7] == -1):
                opponent_diagonal_two++
        elif (board[3] == -1 && board[5] != -1 && board[7] == -1):
                opponent_diagonal_two++
        elif (board[3] == -1 && board[5] == -1 && board[7] == -1):
                loser++

    @staticmethod
    def calculateVertical (board, vertical_one : int, vertical_two: int, opponent_vertical_one: int, opponent_vertical_two: int, winner: int, loser: int):

        for x in range(1, 4):
            if (board[x] == 1 && board[x+3] != 1 && board[x+6] != 1):
                    vertical_one++
            if (board[x] != 1 && board[x+3] == 1 && board[x+6] != 1):
                    vertical_one++
            if (board[x] != 1 && board[x+3] != 1 && board[x+6] == 1):
                    vertical_one++
            if (board[x] == 1 && board[x+3] == 1 && board[x+6] != 1):
                    vertical_two++
            if (board[x] != 1 && board[x+3] == 1 && board[x+6] == 1):
                    vertical_two++
            if (board[x] == 1 && board[x+3] != 1 && board[x+6] == 1):
                    vertical_two++
            if (board[x] == 1 && board[x+3] == 1 && board[x+6] == 1):
                    winner++

            if (board[x] == -1 && board[x+3] != -1 && board[x+6] != -1):
                    opponent_vertical_one++
            if (board[x] != -1 && board[x+3] == -1 && board[x+6] != -1):
                    opponent_vertical_one++
            if (board[x] != -1 && board[x+3] != -1 && board[x+6] == -1):
                    opponent_vertical_one++
            if (board[x] == -1 && board[x+3] == -1 && board[x+6] != -1):
                    opponent_vertical_two++
            if (board[x] != -1 && board[x+3] == -1 && board[x+6] == -1):
                    opponent_vertical_two++
            if (board[x] == -1 && board[x+3] != -1 && board[x+6] == -1):
                    opponent_vertical_two++
            if (board[x] == -1 && board[x+3] == -1 && board[x+6] == -1):
                    loser++

    @staticmethod
    def calculateHorizontal (board, horizontal_one : int, horizontal_two: int, opponent_horizontal_one: int, opponent_horizontal_two: int, winner: int, loser: int):

        digits = [1, 4, 7]
        for x in digits:
            if (board[x] == 1 && board[x+1] != 1 && board[x+2] != 1):
                    horizontal_one++
            if (board[x] != 1 && board[x+1] == 1 && board[x+2] != 1):
                    horizontal_one++
            if (board[x] != 1 && board[x+1] != 1 && board[x+2] == 1):
                    horizontal_one++
            if (board[x] == 1 && board[x+1] == 1 && board[x+2] != 1):
                    horizontal_two++
            if (board[x] != 1 && board[x+1] == 1 && board[x+2] == 1):
                    horizontal_two++
            if (board[x] == 1 && board[x+1] != 1 && board[x+2] == 1):
                    horizontal_two++
            if (board[x] == 1 && board[x+1] == 1 && board[x+2] == 1):
                    winner++

            if (board[x] == -1 && board[x+1] != -1 && board[x+2] != -1):
                    opponent_horizontal_one++
            if (board[x] != -1 && board[x+1] == -1 && board[x+2] != -1):
                    opponent_horizontal_one++
            if (board[x] != -1 && board[x+1] != -1 && board[x+2] == -1):
                    opponent_horizontal_one++
            if (board[x] == -1 && board[x+1] == -1 && board[x+2] != -1):
                    opponent_horizontal_two++
            if (board[x] != -1 && board[x+1] == -1 && board[x+2] == -1):
                    opponent_horizontal_two++
            if (board[x] == -1 && board[x+1] != -1 && board[x+2] == -1):
                    opponent_horizontal_two++
            if (board[x] == -1 && board[x+1] == -1 && board[x+2] == -1):
                    loser++

    #def heuristic(state: np.ndarray):
       #return np.random.randint(1,9)

    #@staticmethod
    #def heuristic2(state: np.ndarray, player: int):
        """ Counts number of our adjacent values in board """

     #   def count_adjacent(array):
      #      return np.count_nonzero(array == 1)

       # def board_value(board, player):
#            board[board == -player] = 0

 #           rows = np.sum([np.sum(board[1:4]), np.sum(board[4:7]), np.sum(board[7:10])])

            # check columns for win board
  #          columns = np.sum([np.sum(board[[1, 4, 7]]), np.sum(board[[2, 5, 8]]), np.sum(board[[3, 6, 9]])])
            # # check diagonals for win board
   #         diagonals = np.sum([np.sum(board[[1, 5, 9]]), np.sum(board[[3, 5, 7]])])
    #        return np.sum([rows, columns, diagonals])

   #     player_1_value = board_value(state, player)
    #    player_2_value = board_value(state, -player)
     #   return player_1_value - player_2_value

