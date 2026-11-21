
# missing mcts import waitng on code


#what im doing is passing game state into neural network -> having the pre trained neural network find the most promising moves and stuff using policy and stuff -> passing that to mcts -> mcts returns selected move to me which i apply to hex and update game

import torch

from trainer import Trainer
# import trainer
from mcts import mcts_search

trainer = Trainer()

# Convert board into tensor for neural network
def encode_board(game):
    board = torch.tensor(game.board)

    player = game.current_player
    opponent = 3 - player


    player_board = (board == player).float()
    opponent_board = (board == opponent).float()
    turn_board = torch.full((11, 11), float(player))

    return torch.stack([player_board, opponent_board, turn_board])


# Create mask for legal moves

def get_legal_mask(game):
    board = torch.tensor(game.board)
    return (board == 0).flatten()


# Get policy and value from neural network
def predict_batch(games):
    """
    For TALIN!!!

    Called by MCTS to evaluate a batch of unexplored leaf game states.

    MCTS should collect non-terminal leaf states and pass them
    as a list of Hex objects. Each object must contain the current
    board and the player whose turn it is.

    The inference engine encodes each board, combines them into
    a batch and runs a single neural network forward pass.
    Illegal moves are masked before returning the probabilities.

    Args:
        games: A list of Hex game state objects.

    Returns:
        policies: A tensor of shape (B, 121), where each row contains
                  the probabilities of playing at each of the 121
                  board positions for a given game state.
                  Illegal moves have a probability of 0.
                  The probability of a move at (row, col) can be
                  accessed using policies[i][row * 11 + col].

        values: A tensor of shape (B,), where each element represents
                how favourable the corresponding game state is for
                the player whose turn it is.
                +1 means a predicted win.
                -1 means a predicted loss.
                 0 means neither player is favoured.

    B is the number of game states passed to the inference engine.
    The predictions are returned in the same order as the input states.

    For each game state, MCTS should:
        1. Use policies[i] to prioritise which moves to explore
           when expanding the leaf node.
        2. Use values[i] to update the estimated values of the
           nodes along the path back to the root, accounting
           for the alternating player perspectives.

    If a game state already has a winner, MCTS should not send
    it to the inference engine. Instead, it should propagate
    the actual game result rather than a predicted value. 
    """
    boards = []
    masks = []

    for game in games:
        boards.append(encode_board(game))
        masks.append(get_legal_mask(game))

    x = torch.stack(boards)
    mask = torch.stack(masks)

    policies, values = trainer.predict(x, mask)

    return policies, values