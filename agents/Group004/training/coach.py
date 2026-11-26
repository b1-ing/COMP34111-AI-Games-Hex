import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from training.Trainer import Trainer
from MCTS import MCTS
from training.ReplayBuffer import ReplayBuffer
log = logging.getLogger(__name__)

def main(cfg, resume=False):
    """
    Performs numIters iterations with numEps episodes of self-play in each
    iteration. After every iteration, it retrains neural network with
    examples in trainExamples (which has a maximum length of maxlenofQueue).
    It then pits the new neural network against the old one and accepts it
    only if it wins >= updateThreshold fraction of games.
    """
    trainer = Trainer(n=cfg.size, channels=cfg.channels, blocks=cfg.blocks)
    buffer = ReplayBuffer(cfg.buffer_positions)
    start = restore(trainer, buffer, cfg) if resume else 0

    for rnd in range(start, cfg.rounds):
        games = run_selfplay(trainer.net, cfg)  # selfplay.py
        buffer.add(games)
        if len(buffer) >= cfg.min_buffer:
            history = trainer.train(*buffer.sample_all(), epochs=cfg.epochs)
            log_csv(cfg, rnd, history)
        save_round(trainer, buffer, cfg, rnd)
        if rnd % cfg.eval_every == 0:
            maybe_promote(trainer, cfg, rnd)


if __name__ == "__main__":
    args = parse_args()                                  # --config, --resume
    main(CONFIGS[args.config], args.resume)
