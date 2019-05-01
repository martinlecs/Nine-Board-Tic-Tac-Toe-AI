import math
import numpy as np
from player.Game import Game
from player.GameTreeNode import GameTreeNode
from player.Heuristic import Heuristic


class AlphaBeta:
    """ Wrapper function for negamax search

    Description of what this function does

    Args:
        node (GameTreeNode) :
        eval_cls (Callable) :
        depth (int) :
        generated_nodes (bool) :

    Returns:
        state, move (numpy.ndarray, int) :

    """

    def __init__(self, node: GameTreeNode, game: Game, eval_cls: Heuristic, depth: int):
        self._node = node
        self._eval_cls = eval_cls
        self._depth = depth
        self._game = game

    def run(self):
        """ Run the minimax search with alpha-beta pruning """

        player = 1
        depth = 0
        self.__alpha_beta(self._node, depth, -math.inf, math.inf, player)
        best_move = max(self._node.children, key=lambda c: c.alpha)

        return best_move.move

    def __alpha_beta(self, node: GameTreeNode, depth: int, alpha: float, beta: float, player: int):
        """ Search game to determine best action; uses negamax implementation and alpha-beta pruning.

        Args:
            node (GameTreeNode) :
            eval_cls (Callable) :
            depth (int) :
            alpha (float) :
            beta (float) :
            player (int) : Can take either 1 or -1

        Returns:
            alpha (float) :

        """
        if self._game.is_terminal(node) or depth == self._depth:
            return self._eval_cls.compute_heuristic(node.state, depth)

        if player == 1:

            best_val = -math.inf

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

