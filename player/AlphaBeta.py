import math
from player.Game import Game
from player.GameTreeNode import GameTreeNode
from player.Heuristic import Heuristic


class AlphaBeta:
    """ Wrapper function for Alpha Beta Search.

    Standard minimax search with alpha beta pruning optimisation.

    Attributes:
        node (GameTreeNode): The root node.
        game (Game): Game class instance.
        eval_cls (Heuristic): Heuristic class instance.
        depth (int): Depth to run search to.

    """

    def __init__(self, node: GameTreeNode, game: Game, eval_cls: Heuristic, depth: int):
        self._node = node
        self._game = game
        self._eval_cls = eval_cls
        self._depth = depth

    def run(self):
        """ Run the minimax search with alpha-beta pruning

        Returns:
              Int that represents the best move found.

        """
        player = 1  # Assume that we are player
        depth = 0   # start at depth 0 and increment to desired depth as search continues
        self.__alpha_beta(self._node, depth, -math.inf, math.inf, player)
        best_move = max(self._node.children, key=lambda c: c.alpha)
        return best_move.move

    def __alpha_beta(self, node: GameTreeNode, depth: int, alpha: float, beta: float, player: int):
        """ Search game to determine best action; uses negamax implementation and alpha-beta pruning.

        Args:
            node (GameTreeNode): The root or parent node.
            depth (int): The depth of the current search.
            alpha (float): The best value found for current player.
            beta (float): The best value found for the opponent.
            player (int) : Can take either 1 or -1 (Current player == 1 and Opponent == -1)

        Returns:
            A number (float) representing the best move possible for the player.

        """
        if self._game.is_terminal(node) or depth == self._depth:
            return self._eval_cls.compute_heuristic(node.state, depth)

        if player == 1:

            best_val = -math.inf

            # generate children and recursively apply the alpha beta search to each child
            for child in self._game.generate_moves(node.state, node.get_board_num(), player):

                ret_val = self.__alpha_beta(child, depth + 1, alpha, beta, -player)
                best_val = max(best_val, ret_val)
                alpha = max(alpha, best_val)

                # We only need keep track of the children generated right below the root
                # so that we can find the best move
                if depth == 0:
                    node.children.append(child)
                    child.alpha = ret_val

                # we can prune on this condition
                if beta <= alpha:
                    return best_val

            return best_val

        else:

            best_val = math.inf

            for child in self._game.generate_moves(node.state, node.get_board_num(), player):
                best_val = min(best_val, self.__alpha_beta(child, depth + 1, alpha, beta, -player))
                beta = min(beta, best_val)

                if beta <= alpha:
                    return best_val

            return best_val

