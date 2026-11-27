"""Plain UCT Monte Carlo Tree Search for Hex (no heuristics, random rollouts).

Works against the Hex class in game_state.py, using only:
    clone(), make_move(cell), winner, current_player, board, SIZE

Each iteration runs four phases:
    select -> expand -> rollout -> backpropagate
and the move played is the root child with the most visits.

THIS REALLY SHOULD BE IN C++
"""
import math
import random

from agents.Group004.game_state import Hex


def legal_moves(state):
    """All empty cells, or [] if the game is over."""
    if state.winner is not None:
        return []
    return [(r, c) for r in range(state.SIZE) for c in range(state.SIZE)
            if state.board[r][c] is None]


class Node:
    """A position in the search tree, reached by `move` played by `player`."""

    def __init__(self, parent, move, player, state):
        self.parent = parent
        self.move = move                    # the move that led here
        self.player = player                # who played it (None at root)
        self.children = []
        self.untried = legal_moves(state)   # moves not yet expanded
        self.N = 0                          # visit count
        self.W = 0.0                        # wins for self.player


def ucb(child, c=1.41):
    """UCB1 score: win rate (exploit) + bonus for rarely visited moves (explore)."""
    exploit = child.W / child.N
    explore = c * math.sqrt(math.log(child.parent.N) / child.N)
    return exploit + explore


def select(node, state, c):
    """Descend through fully expanded nodes, always taking the best UCB child.

    W is stored from the mover's perspective, so `max` picks the best move for
    whichever player is choosing at each level.
    """
    while not node.untried and node.children:
        node = max(node.children, key=lambda ch: ucb(ch, c))
        state.make_move(node.move)
    return node


def expand(node, state):
    """Add one untried move as a new child. Terminal nodes are returned as-is."""
    if not node.untried:
        return node
    move = node.untried.pop(random.randrange(len(node.untried)))
    player = state.current_player  # read before make_move; it won't switch on a win
    state.make_move(move)
    child = Node(node, move, player, state)
    node.children.append(child)
    return child


def rollout(state):
    """Play uniformly random moves until someone wins. Returns the winner."""
    while state.winner is None:
        state.make_move(random.choice(legal_moves(state)))
    return state.winner


def backpropagate(node, winner):
    """Credit the result to every node on the path back to the root."""
    while node is not None:
        node.N += 1
        if node.player == winner:
            node.W += 1
        node = node.parent


def best_move(root):
    """Most-visited root child: more robust than highest win rate."""
    return max(root.children, key=lambda ch: ch.N).move


def mcts_search(root_state, iters=1000, c=1.41, root=None):
    """Run UCT from root_state and return (move, root).

    Pass a previous `root` (see advance_root) to reuse its subtree's statistics.
    """
    if root is None:
        root = Node(None, None, None, root_state)
    for _ in range(iters):
        state = root_state.clone()   # scratch board; the real game is untouched
        node = select(root, state, c)
        node = expand(node, state)
        winner = rollout(state)
        backpropagate(node, winner)
    return best_move(root), root


def advance_root(root, *moves):
    """Follow `moves` down the tree and make that node the new root.

    Returns None if any move leads somewhere the search never explored.
    """
    for move in moves:
        if root is None:
            return None
        root = next((ch for ch in root.children if ch.move == move), None)
    if root is not None:
        root.parent = None
    return root


class RandomPlayer:
    """Trivial adversary for testing."""

    def choose(self, game):
        return random.choice(legal_moves(game))


def play(adversary, bot_player=1, iters=1000, reuse_tree=True, verbose=True):
    """Play one game of the MCTS bot against `adversary`. Returns the winner."""
    game = Hex()
    root = None
    my_move = None
    while game.winner is None:
        if game.current_player == bot_player:
            move, root = mcts_search(game, iters, root=root)
            my_move = move
        else:
            move = adversary.choose(game)
            if reuse_tree and root is not None and my_move is not None:
                root = advance_root(root, my_move, move)
            else:
                root = None
        mover = game.current_player
        game.make_move(move)
        if verbose:
            print(f"player {mover} -> {move}")
    if verbose:
        print(f"winner: player {game.winner}")
    return game.winner


if __name__ == "__main__":
    play(RandomPlayer(), bot_player=1, iters=500)
