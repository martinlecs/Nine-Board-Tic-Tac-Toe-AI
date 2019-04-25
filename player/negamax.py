import math
from player.GameTreeNode import GameTreeNode
from typing import Callable


def minimax(node: GameTreeNode, eval_cls: Callable, depth: int, generated_nodes=False):
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

    player = 1
    node.generate_moves(player)

    best_move = max(node.children, key=lambda m: negamax(m, eval_cls, depth - 1, -math.inf, math.inf, player))
    print([i.alpha for i in node.children])

    if generated_nodes:
        return best_move.state, best_move.move, node.all_generated_nodes()
    return best_move.state, best_move.move


def negamax(node: GameTreeNode, eval_cls: Callable, depth: int, alpha: float, beta: float, player: int):
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

    if node.is_terminal_node() or depth == 0:
        return eval_cls(node.state).heuristic()

    node.generate_moves(-player)
    for child in node.children:
        alpha = max(alpha, -negamax(child, eval_cls, depth - 1, -beta, -alpha, -player))
        node.alpha = alpha
        if alpha >= beta:
            return alpha
    node.alpha = alpha
    return alpha
