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
        self._players = []
        self._nodes_generated = 0

    @property
    def players(self):
        return self._players

    @property
    def nodes_generated(self):
        return self._nodes_generated

    def run(self):
        """ Run the minimax search with alpha-beta pruning """
        player = 1
        print(self.__alpha_beta(self._node, self._eval_cls, self._depth, -math.inf, math.inf, player))
        best_move = max(self._node.children, key=lambda c: c.alpha)
        # print(best_move.state)
        # print()

        return best_move.move

    def __alpha_beta(self, node: GameTreeNode, eval_cls: Callable, depth: int, alpha: float, beta: float, player: int):
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
            return node.heuristic_val

        if player == 1:
            bestVal = -math.inf

            node.generate_moves(player)
            # self._nodes_generated += len(node.children)
            # self._players.append(player)

            for child in node.children:
                bestVal = max(bestVal, self.__alpha_beta(child, eval_cls, depth - 1, alpha, beta, -player))
                alpha = max(alpha, bestVal)
                child.alpha = bestVal
                if beta <= alpha:
                    return bestVal
            return bestVal

        else:

            node.generate_moves(player)
            # self._nodes_generated += len(node.children)
            # self._players.append(player)
            bestVal = math.inf

            for child in node.children:
                bestVal = min(bestVal, self.__alpha_beta(child, eval_cls, depth - 1, alpha, beta, -player))
                beta = min(beta, bestVal)
                child.beta = bestVal
                if beta <= alpha:
                    return bestVal
            return bestVal
