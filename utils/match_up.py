import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

from typing import List
import chess
from utils.general import read_yaml
from core.torch_models import ChessAgent
from IPython.display import display
from utils.chess_env import ChessEnv, save_recording

def head_to_head_match(white_config_name: str, black_config_name: str, initial_state: str = None,
                       step_limit: int = 250, record_path: str = None) -> List[chess.Move]:
    """
    Plays a head-to-head match between 2 computer agents and returns the move stack of their game.

    :param white_config_name: The name of a config file to play the white pieces.
    :param black_config_name: The name of a config file to play the black pieces.
    :param initial_state: A FEN board encoding of the starting state of the game. If not provided, the game
        begins at the standard chess start.
    :param step_limit: Sets and upper bound for the max number of moves allowed.
    :param record_path: If provided, then a recording of the match will be saved to the file path specified.
    :return: Returns the move stack of the game.
    """
    assert record_path.endswith(".mp4"), "record_path must end with the .mp4 file extension"
    model_w = ChessAgent(read_yaml(os.path.join(PARENT_DIR, f"config/{white_config_name}.yml")))
    model_b = ChessAgent(read_yaml(os.path.join(PARENT_DIR, f"config/{black_config_name}.yml")))

    env = ChessEnv(initial_state=initial_state, step_limit=step_limit)
    while not env.ep_ended:  # Play until the game is finished
        if env.board.turn: # White to move next
            action, _, _ = model_w.get_best_action(env.board.fen())
            next_state, reward, terminated, truncated = env.step(action)
        else: # Black to move next
            action, _, _ = model_b.get_best_action(env.board.fen())
            next_state, reward, terminated, truncated = env.step(action)

    if record_path is not None:
        save_recording(env.board.move_stack, output_path=record_path)

    # Return the move_stack of the game along with the outcome of the game (if any), otherwise the outcome
    # is that the game was truncated at step_limit
    return env.board.move_stack, env.board.outcome()


def interactive_match(config_name: str, player_color: str = "white", initial_state: str = None,
                      verbose: bool = False) -> None:
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
    :param initial_state: A FEN board encoding of the starting state of the game. If not provided, the game
        begins at the standard chess start.
    :param verbose: If set to True, then a verbose print out of the agent decision making is provided.
    :return: None.
    """
    player_color = player_color.lower()
    assert player_color.lower() in ["white", "black"], "Player color must be either white or black"
    player_color = chess.WHITE if player_color == "white" else chess.BLACK

    config = read_yaml(os.path.join(PARENT_DIR, f"config/{config_name}.yml"))
    model = ChessAgent(config)
    board = chess.Board(initial_state) if initial_state is not None else chess.Board()
    check = board.king(board.turn) if board.is_check() else None
    display(chess.svg.board(board, orientation=player_color, check=check))

    while not board.is_game_over():  # Play until the game is finished
        print(board.fen()) # Show the FEN of the game on each turn before the next move is made
        if board.turn == player_color:
            move_valid = False
            while not move_valid:
                try:
                    player_move_san = input("Input move in SAN: ")
                    if player_move_san == "quit":  # Exit immediately if the user enters "quit"
                        return None
                    elif player_move_san == "undo" and len(board.move_stack) >= 2:
                        # Undo the last 2 moves to get back to your last move
                        board.pop()  # Undo opponent's last move
                        board.pop()  # Undo player's last move before that
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
        else:  # Otherwise it's the turn of the AI chess bot RL agent to play
            if verbose is False:
                move = model.agent_move_func(board)

            else: # Verbose printing, show what the model was thinking on this play
                best_action, state_value, action_values, info = model._search_func(
                    state=board.fen(), model=model.v_network, **model.config["search_func"])
                legal_moves = list(board.legal_moves)
                print(f"\nState Value: {state_value:.2f}")
                print(f"Best Action: {best_action} {legal_moves[best_action]}")
                print(f"Nodes Evaluated: {info[0]}, Max Depth: {info[1]}, Terminal Nodes: {info[2]}")
                move_vals = [(move, val) for move, val in zip(legal_moves, action_values)]
                # List each possible move and it's approx value in descending order
                for move, value in sorted(move_vals, key=lambda x: -x[1]):
                    print(f"   {move} {value:.3f}")
                move = legal_moves[best_action]

            board.push(move)
        check = board.king(board.turn) if board.is_check() else None
        display(chess.svg.board(board, orientation=player_color, lastmove=board.peek(), check=check))

    print(board.fen()) # Report the FEN of the game on the last move as well
    outcome = board.outcome()
    msg = ("- black wins!" if board.turn else "- white wins!") if board.is_checkmate() else ""
    print(f"Outcome: {outcome.termination.name} {msg}")

if __name__ == "__main__":
    interactive_match("mlp_agent", "white", verbose=True) # Play against one of the AI models
    # interactive_match("heuristic_agent", "white", verbose=True) # Play against one of the AI models
    # Have 2 models play each other head-to-head
    # head_to_head_match("CNN_agent", "CNN_agent", record_path=os.path.join(PARENT_DIR, "test.mp4"))
