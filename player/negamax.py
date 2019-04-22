import math
from typing import Callable

from player.GameTreeNode import GameTreeNode


def minimax(node: GameTreeNode, eval_fn: Callable, depth: int, generated_nodes=False):
    """ Wrapper function for negamax search """

    player = 1
    node.generate_moves(player)
    best_move = max(node.children, key=lambda m: negamax(m, eval_fn, depth - 1, -math.inf, math.inf, player))

    if generated_nodes:
        return best_move.state, best_move.move, node.all_generated_nodes()
    return best_move.state, best_move.move


def negamax(node: GameTreeNode, eval_fn: Callable, depth: int, alpha: float, beta: float, player):
    """ Search game to determine best action; uses negamax implementation and alpha-beta pruning. """

    if node.is_terminal_node() or depth == 0:
        return eval_fn(node.board)

    node.generate_moves(-player)
    for child in node.children:
        alpha = max(alpha, -negamax(child, eval_fn, depth - 1, -beta, -alpha, -player))
        if alpha >= beta:
            return alpha
    return alpha
