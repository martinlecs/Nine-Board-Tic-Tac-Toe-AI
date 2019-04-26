import math
from player.GameTreeNode import GameTreeNode
from typing import Callable


class minimax:
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

    def __init__(self, node: GameTreeNode, eval_cls: Callable, depth: int, generated_nodes=False):
        self._node = node
        self._eval_cls = eval_cls
        self._depth = depth
        self._generated_nodes = generated_nodes

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
        player = 1
        # self.__negamax(self._node, self._eval_cls, self._depth, -math.inf, math.inf, player)

        # print(self.__alpha_beta(self._node, self._eval_cls, self._depth, player))

        self.__minimax(self._node, self._eval_cls, self._depth, -math.inf, math.inf, player)
        best_move = max(self._node.children, key=lambda c: c.alpha)

        # self._node.generate_moves(player)
        # best_move = max(self._node.children, key=lambda m: self.__alpha_beta(m, self._eval_cls, self._depth, player))

        if self._generated_nodes:
            return best_move.state, best_move.move, self._node.all_generated_nodes()
        return best_move.state, best_move.move

    def __negamax(self, node: GameTreeNode, eval_cls: Callable, depth: int, alpha: float, beta: float, player: int):
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
            node.alpha = eval_cls(node.state).heuristic()
            return node.alpha

        node.generate_moves(player)
        self._nodes_generated += len(node.children)
        self._players.append(player)

        for child in node.children:
            alpha = max(alpha, -self.__negamax(child, eval_cls, depth - 1, -beta, -alpha, -player))
            child.alpha = alpha
            if alpha >= beta:
                return alpha
        # node.alpha = alpha
        return alpha

    def __alpha_beta(self, node: GameTreeNode, eval_cls: Callable, depth: int, player: int):

        def max_value(node, alpha, beta, depth, player):

            if node.is_terminal_node(node.state) or depth == 0:
                node.alpha = eval_cls(node.state).heuristic()
                return node.alpha

            v = -math.inf
            node.generate_moves(player)
            self._nodes_generated += len(node.children)
            self._players.append(player)

            for child in node.children:
                v = max(v, min_value(child, alpha, beta, depth - 1, -player))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(node, alpha, beta, depth, player):

            if node.is_terminal_node(node.state) or depth == 0:
                return eval_cls(node.state).heuristic()

            v = math.inf
            node.generate_moves(player)
            self._nodes_generated += len(node.children)
            self._players.append(player)

            for child in node.children:
                v = min(v, max_value(child, alpha, beta, depth - 1, -player))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        return max_value(node, -math.inf, math.inf, depth, player)
        # return min_value(node, -math.inf, math.inf, depth, player)

    #TODO: consider using a bool for player variable instead of an int
    def __minimax(self, node: GameTreeNode, eval_cls: Callable, depth: int, alpha: float, beta: float, player: int):

        if node.is_terminal_node(node.state) or depth == 0:
            node.alpha = eval_cls(node.state).heuristic()
            return node.alpha
            # return eval_cls(node.state).heuristic()

        if player == 1:
            bestVal = -math.inf

            node.generate_moves(player)
            self._nodes_generated += len(node.children)
            self._players.append(player)

            for child in node.children:
                value = self.__minimax(child, eval_cls, depth - 1, alpha, beta, -player)
                bestVal = max(bestVal, value)
                alpha = max(alpha, bestVal)
                child.alpha = bestVal
                if beta <= alpha:
                    break
            return bestVal

        else:

            node.generate_moves(player)
            self._nodes_generated += len(node.children)
            self._players.append(player)
            bestVal = math.inf

            for child in node.children:
                value = self.__minimax(child, eval_cls, depth - 1, alpha, beta, -player)
                bestVal = min(bestVal, value)
                beta = min(beta, bestVal)
                if beta <= alpha:
                    return bestVal
            return bestVal
