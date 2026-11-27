from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import torch

from agents.Group004.game_state import Hex
from agents.Group004.mcts import mcts_search
from agents.Group004.inference import predict_batch

class NaiveAgent(AgentBase):
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.

    The class inherits from AgentBase, which is an abstract class.
    The AgentBase contains the colour property which you can use to get the agent's colour.
    You must implement the make_move method to make the agent functional.
    You CANNOT modify the AgentBase class, otherwise your agent might not function.
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will alwhat so change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent's move
        """

        if turn == 2:
            if self.should_swap(board):
                return Move(-1, -1)

        game = self.convert_board(board)

        move, root = mcts_search(game, iters=1000)

        return Move(*move)
    
    # need to tweak still not confirmed
    def should_swap(self, game, nnet):
        policies, values = predict_batch([game], nnet)

        value = values[0].item()

        return value > 0.3
    
    def convert_board(self, board):
        game = Hex()

        for row in board.tiles:
            for tile in row:
                if tile.colour is not None:
                    game.current_player = tile.colour
                    game.make_move((tile.x, tile.y))

        game.current_player = self.colour
        game.swap_available = (
            sum(tile.colour is not None
                for row in board.tiles
                for tile in row) == 1
            and self.colour == Colour.BLUE
        )

        return game