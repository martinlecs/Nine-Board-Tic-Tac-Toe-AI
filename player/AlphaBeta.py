import math
from player.GameTreeNode import GameTreeNode
from typing import Callable


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

    def __init__(self, node: GameTreeNode, eval_cls: Callable, depth: int):
        self._node = node
        self._eval_cls = eval_cls
        self._depth = depth

        # Debugging
        self._nodes_generated = 0

    @property
    def nodes_generated(self):
        return self._nodes_generated

    def run(self):
        """ Run the minimax search with alpha-beta pruning """

        player = 1
        self.__alpha_beta(self._node, self._depth, -math.inf, math.inf, player)
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
        # TODO: Fix is_terminal_node to handle draws
        if node.is_terminal_node(node.state) or depth == 0:
            return self._eval_cls.compute_heuristic(node.state, self._depth - depth)

        if player == 1:
            bestVal = -math.inf

            node.generate_moves(player)
            self._nodes_generated += len(node.children)

            for child in node.children:
                bestVal = max(bestVal, self.__alpha_beta(child, depth - 1, alpha, beta, -player))
                alpha = max(alpha, bestVal)
                child.alpha = bestVal
                if beta <= alpha:
                    return bestVal
            return bestVal

        else:

            node.generate_moves(player)
            self._nodes_generated += len(node.children)
            bestVal = math.inf

            for child in node.children:
                bestVal = min(bestVal, self.__alpha_beta(child, depth - 1, alpha, beta, -player))
                beta = min(beta, bestVal)
                child.beta = bestVal
                if beta <= alpha:
                    return bestVal
            return bestVal
