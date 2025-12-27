import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import chess
from utils.general import read_yaml
from utils.chess_env import ChessEnv
from core.torch_models import ChessAgent
from IPython.display import display, clear_output

def interactive_match(config_name: str, player_color: str = "white", state: str = None) -> None:
    """
    This function allows a user to play an interactive match vs one of the RL chess agents saved to disk in
    the results folder. Players will be prompted to input their next move and the RL chess agent will play
    the other side of the board until the game is over.

    Moves are expected to be entered in standard algebraic notation (SAN) e.g. e4, Qh7, g1g5 etc.
    If "quit" is input, the game session ends. If "undo" is entered, the last 2 moves are reversed i.e. the
    AI's last move and the player's prior move. Games run until an end game condition is met e.g. checkmate,
    stalemate, insufficient material, repetition etc.

    :param config_name: The config name of the chess agent to play against e.g. "cnn_agent".
    :param player_color: The color of the user playing i.e. "white" or "black".
    :return: None.
    """
    player_color = player_color.lower()
    assert player_color.lower() in ["white", "black"], "Player color must be either white or black"
    player_color = chess.WHITE if player_color == "white" else chess.BLACK

    config = read_yaml(os.path.join(PARENT_DIR, f"config/{config_name}.yml"))
    model = ChessAgent(config)
    board = chess.Board(state) if state is not None else chess.Board()
    check = board.king(board.turn) if board.is_check() else None
    display(chess.svg.board(board, orientation=player_color, check=check))

    while not board.is_game_over(): # Play until the game is finished
        if board.turn == player_color:
            move_valid = False
            while not move_valid:
                try:
                    player_move_san = input("Input move in SAN: ")
                    if player_move_san == "quit": # Exit immediately if the user enters "quit"
                        return None
                    elif player_move_san == "undo" and len(board.move_stack) >= 2:
                        # Undo the last 2 moves to get back to your last move
                        board.pop() # Undo opponent's last move
                        board.pop() # Undo player's last move before that
                        if len(board.move_stack) > 0:
                            check = board.king(board.turn) if board.is_check() else None
                            display(chess.svg.board(board, orientation=player_color, lastmove=board.peek(),
                                                    check=check))
                        else:
                            display(chess.svg.board(board, orientation=player_color))
                    else:  # All other inputs are interpreted as moves
                        board.push_san(player_move_san)
                        move_valid = True
                except:
                    print("Move invalid, please try again")
        else: # Otherwise it's the turn of the AI chess bot RL agent to play
            move = model.agent_move_func(board)
            board.push(move)
        check = board.king(board.turn) if board.is_check() else None
        display(chess.svg.board(board, orientation=player_color, lastmove=board.peek(), check=check))

    outcome = board.outcome()
    msg = ("- black wins!" if board.turn else "- white wins!") if board.is_checkmate() else ""
    print(f"Outcome: {outcome.termination.name} {msg}")


if __name__ == "__main__":
    interactive_match("mlp_agent", "white")
