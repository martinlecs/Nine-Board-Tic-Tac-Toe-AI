from player.GameTreeNode import GameTreeNode
from typing import Callable
import math


def heuristic():
    return 1


def minimax(node: GameTreeNode, depth: int):
    """ Wrapper function for negamax search """
    return negamax(node, heuristic, depth, -math.inf, math.inf)


def negamax(node: GameTreeNode, eval_fn: Callable, depth: int, alpha: float, beta: float):
    """ Search game to determine best action; uses negamax implementation and alpha-beta pruning. """
    if node.is_terminal_node() or depth == 0:
        return eval_fn(node.get_state())

    node.generate_moves()   # will have to keep track of which player I am when generating moves
    node.order_moves()  # can be optimised internally inside the class
    for child in node.get_children():
        alpha = max(alpha, -negamax(child, depth - 1, -beta, -alpha))
        if alpha >= beta:
            return alpha
    return alpha


    # where is the action taken, where is the new state?
