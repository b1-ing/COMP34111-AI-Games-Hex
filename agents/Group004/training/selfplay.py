import numpy as np
from agents.GroupXYZ.game import Hex, RED, BLUE
from agents.GroupXYZ.mcts import search, advance

def visits_to_pi(visits, player, n):
    pi = np.zeros((n, n), np.float32)
    for (x, y), count in visits.items():
        pi[x, y] = count
    if player == BLUE:
        pi = pi.T                                  # same frame as encode()
    return (pi / pi.sum()).reshape(-1)

def rotate180(board, pi, n):
    return board[::-1, ::-1].copy(), pi.reshape(n, n)[::-1, ::-1].reshape(-1).copy()

def play_one_game(evaluate, cfg, rng):
    game, root, history = Hex(cfg.size), None, []
    while game.winner is None:
        visits, root = search(game, evaluate, sims=cfg.sims, noise=True, root=root)
        player = game.current_player
        history.append((np.array(game.board, np.int8), player,
                        visits_to_pi(visits, player, cfg.size)))
        moves, counts = zip(*visits.items())
        if len(history) <= cfg.temp_moves:         # explore early
            move = moves[rng.choice(len(moves), p=np.array(counts) / sum(counts))]
        else:                                      # then play the best move
            move = moves[int(np.argmax(counts))]
        game.make_move(move)
        root = advance(root, move)

    out = []
    for board, player, pi in history:
        z = 1 if player == game.winner else -1
        out.append((board, player, pi, z))
        rb, rpi = rotate180(board, pi, cfg.size)
        out.append((rb, player, rpi, z))           # 180° rotation keeps who connects where
    return out

def run_selfplay(evaluate, cfg, games, seed=0):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(games):
        data += play_one_game(evaluate, cfg, rng)
    return data