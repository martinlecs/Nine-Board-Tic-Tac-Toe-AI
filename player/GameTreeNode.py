import numpy as np


class GameTreeNode:

    def __init__(self, state: np.ndarray, current_board: int, parent=None):
        self._state = state
        self._current_board = current_board
        self._parent = parent
        self._children = None

    def is_terminal_node(self):
        # check if win state
        pass

    def generate_moves(self):
        """ Generates all possible moves for current player by looking at empty squares as potential moves """
        board = self._state[self._current_board]
        move_list = []
        for i in range(1, len(board)):
            if board[i] == 0:
                new_move = np.copy(board)
                new_move[i] = 1
                move_list.append(new_move)
                continue
        self._children = np.asarray(move_list)

    def order_moves(self):
        # need to use heuristic on heuristic
        # possibly order moves based on which board has best position
        pass

    def get_children(self):
        return self._children

    def get_size_children(self):
        return len(self._children)

    def get_state(self):
        return self._state


if __name__ == "__main__":
    g = GameTreeNode(np.zeros((10, 10), dtype="int8"), 5)
    g.generate_moves()
    print(g.get_children())
