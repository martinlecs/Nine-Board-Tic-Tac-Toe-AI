from player.GameState import GameState


class GameTreeNode:

    def __init__(self, state: GameState, parent=None):
        self._state = state
        self._parent = parent
        self._children = []

    def is_terminal_node(self):
        pass

    def generateMoves(self):
        pass

    def orderMoves(self):
        pass
