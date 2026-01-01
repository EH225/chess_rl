"""
This module creates a simple interface for interacting with a chess game environment.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import numpy as np
from typing import Tuple, Dict, Union, List
import os, chess, cv2, io, chess.svg, cairosvg
from PIL import Image


def move_stack_to_states(move_stack: List[chess.Move], state: str = None) -> List[str]:
    """
    Takes in a sequential list of moves (move_stack) and plays them out from an initial starting board state
    and records as a list of strings all the intermediate states reached as FEN encodings.

    :param move_stack: A stack of chess.Moves that are legal.
    :param state: A starting state from which the move stack evolves.
    :return: A list of FEN state encodings for each state reached during play.
    """
    board = chess.Board() if state is None else chess.Board().state
    states = [board.fen(), ]
    for move in move_stack:
        board.push(move)
        states.append(board.fen())
    return states


def material_diff(state: str) -> float:
    """
    Computes the net material difference of the current board state from the perpsective of the player who is
    to move next where each piece is worth:
        pawn (1), knight (3), bishop (3), rook (5), queen (9), king (1)

    :param state: A FEN string denoting the current game state.
    :return: A int value representing the net material difference for the player whose turn it is to go next.
    """
    piece_values = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 1}
    board = chess.Board(state)  # Convert to a chess.Board object to access the piece map
    net_material = 0
    for p in board.piece_map().values():
        piece_val = piece_values[p.symbol().lower()]  # Get the absolute value of each piece
        if board.turn is True and p.symbol().islower():  # For white, lower-case pieces are foes
            piece_val *= (-1)
        elif board.turn is False and p.symbol().isupper():  # For black, upper-case pieces are foes
            piece_val *= (-1)
        net_material += piece_val

    return net_material


def relative_material_diff(state: str) -> float:
    """
    Computes a heuristic estimate of the game outcome [-1, +1] using the relative material difference from
    the perspective of the player who is to go next where the value of each piece is:
        pawn (0.125), knight (3), bishop (3), rook (5), queen (10), king (1)

    The relative material difference is defined as:
        (player material) / (total material) * 2 - 1 -> maps to a value [-1, +1]

    If the current player to move is in check, an additional -0.1 penalty is added to discentivize being in
    check. Check is a vulnerable position and restricts the moves of the player moving next. This heuristic
    function also return -1 or 0 if the game is at a terminal state (was checkmated or reached a draw). All
    values returned are clipped to [-1, +1].

    :param state: A FEN string denoting the current game state.
    :return: A float value estimate of the game outcome based on relative material.
    """
    board = chess.Board(state)  # Convert to a chess.Board object to access the piece map
    if board.is_game_over():  # If the game is in an end state, return the terminal reward
        return -1.0 if board.is_checkmate() else 0.0

    piece_values = {"p": 0.125, "n": 3, "b": 3, "r": 5, "q": 10, "k": 1}
    total_material, player_material = 0, 0
    for p in board.piece_map().values():
        piece_val = piece_values[p.symbol().lower()]  # Get the absolute value of each piece
        if board.turn is True and p.symbol().isupper():  # For white, upper-case pieces are friendly
            player_material += piece_val
        elif board.turn is False and p.symbol().islower():  # For black, lower-case pieces are friendly
            player_material += piece_val
        total_material += piece_val

    return np.clip((player_material / total_material) * 2 - 1 + (-0.1 if board.is_check() else 0), -1, 1)


def create_ep_record(move_stack: List[chess.Move], initial_state: str = None) -> Dict:
    """
    This method takes in a move_stack of legal moves from an original board starting position and runs
    them through the chess game env and records key info about what occured during it along the way i.e.
    what the outcome was, how many checks were made by each side, pawn promotions etc.

    If the game has not reached a terminal state by the end of move_stack, then the outcome will be
    recorded as Truncated.

    :param move_stack: A list of chess.Moves objects denoting the evolution of the game.
    :return: A dictionary with key info from the progression of the game recorded in move_stack.
    """
    board = chess.Board() if initial_state is None else chess.Board(initial_state)
    ep_df_cols = ["episode_id", "outcome", "winner", "total_moves", "white_material_diff", "white_checks",
                  "black_checks", "white_promotions", "black_promotions", "white_en_passant",
                  "black_en_passant", "white_ks_castle", "black_ks_castle", "white_qs_castle",
                  "black_qs_castle", "end_state"]
    ep_record = {col: 0 for col in ep_df_cols}  # Record key info in a dictionary

    for move in move_stack:  # Re-play the game again, apply each move
        color = "white" if board.turn is chess.WHITE else "black"  # Color of the player to move next

        # Record key info about the move that is about to be made
        if board.is_en_passant(move):  # Check if this move is an en passant move
            ep_record[f"{color}_en_passant"] += 1
        if board.is_kingside_castling(move):  # Check if this move is a king-side castle
            ep_record[f"{color}_ks_castle"] += 1
        if board.is_queenside_castling(move):  # Check if this move is a queen-side castle
            ep_record[f"{color}_qs_castle"] += 1
        if board.gives_check(move):  # Check if the move creates a check
            ep_record[f"{color}_checks"] += 1
        if move.promotion:  # Check if this move is a pawn promotion
            ep_record[f"{color}_promotions"] += 1

        board.push(move)  # Make the next move in the sequence of moves to update the state

    outcome = board.outcome()
    if outcome:  # Check if the game has ended after the last move was made
        ep_record["outcome"] = outcome.termination.name
        if outcome.winner is chess.WHITE:
            ep_record["winner"] = "white"
        elif outcome.winner is chess.BLACK:
            ep_record["winner"] = "black"
        else:
            ep_record["winner"] = "none"
    else:  # Otherwise record the outcome as Truncated if the game hasn't yet ended
        ep_record["outcome"] = "TRUNCATED"
        ep_record["winner"] = "none"

    sign_flip = (1 if board.turn == chess.WHITE else 1)
    ep_record["white_material_diff"] = material_diff(board.fen()) * sign_flip
    ep_record["total_moves"] = len(move_stack)
    ep_record["end_state"] = board.fen()  # Record the ending state FEN
    return ep_record


def board_svg_to_arr(board: chess.Board) -> np.ndarray:
    """
    This function takes in a chess.Board object and converts the svg image representation to a np.ndarray
    of pixel values for writing frames to a mp4 video file.

    :param board: The input chess.Board from which to extract the svg image of the current board.
    :return: A np.ndarray of the current board including highlights and board annotations.
    """
    svg = board._repr_svg_()  # Extract the current board svg with annotations and overlays
    png_bytes = cairosvg.svg2png(bytestring=svg.encode())
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return np.array(img)


def save_recording(move_stack: List[chess.Move], output_path: str, fps: int = 1) -> None:
    """
    Saves a recording of a game which evolves through time according to move_stack, a python list of moves
    that can be executed in a chess.Board env.

    :param move_stack: A list of chess.Move objects describing the evolution of the game.
    :param output_path: The output path to write the recording to ending in .mp4.
    :param fps: Specify a frames-per-second for the output video, a default of 1 second per move is standard.
    :return: None, writes to output_dir instead.
    """
    board = chess.Board()  # Init a new Board obj to play through the moves

    # Pre-render one frame to get the height and width dims
    frame = board_svg_to_arr(board)  # Convert from svg to a np array of pixels
    h, w, c = frame.shape

    # Create video writer (MP4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    try:
        # Write initial board position
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Replay all the moves of the game in order and write each frame to the output video
        for move_uci in move_stack:
            board.push(move_uci)
            out.write(cv2.cvtColor(board_svg_to_arr(board), cv2.COLOR_RGB2BGR))
    except Exception as e:
        print("Exception occurred", e)

    out.release()  # Make sure to release even if there is an error
    print(f"Recording written to: {output_path}")


def save_move_stack(move_stack: List[chess.Move], output_path: str) -> None:
    """
    Saves a sequence of moves contained in move_stack to a text file.

    :param move_stack: A list of chess.Move objects describing the evolution of the game.
    :param output_path: The output path to write the recording to ending in .txt.
    """
    output_str = " ".join([move.xboard() for move in move_stack])
    try:
        with open(output_path, "w") as file:
            file.write(output_str)
    except IOError as e:
        print(f"An error occurred: {e}")


class ActionSpace:
    def __init__(self, board: chess.Board):
        """
        An action space derived from the current board position.
        """
        self.actions = list(board.legal_moves)  # Cast to a list
        self.n = len(self.actions)  # Record how many legal moves there are from the current position

    def sample(self) -> int:
        """
        Randomly samples an index of an action from the action space of legal moves.
        """
        return np.random.randint(0, self.n)


class ChessEnv:
    """
    Env that wraps around a chess.Board() object instance to give familiar methods and attributes including
    self.step(), self.reset(), and self.action_space for RL agent interactions.
    """

    def __init__(self, step_limit: int = 250, record_dir: str = None, initial_state: str = None):
        self.step_limit = step_limit  # Record the episodic step count limit after which we truncate
        if initial_state is not None:  # If specified, start the env in the initial state passed (a FEN str)
            self.board = chess.Board(initial_state)
        else:  # Otherwise, start off with the default starting arrangement
            self.board = chess.Board()
        self.action_space = ActionSpace(self.board)  # Init an action space based on the current board
        self.episode_id = 0  # Keep track of the number of episodes elapsed
        self.step_count = 0  # Keep track of the step count within the current episode
        self.record = False  # Set a toggle for whether to record board states or not
        self.record_dir = record_dir if record_dir is not None else os.path.join(PARENT_DIR, "recordings/")
        os.makedirs(self.record_dir, exist_ok=True)  # Make the recordings save directory if needed
        self.ep_ended = self.board.is_game_over()  # Keep track of if the episode has now ended

    def reset(self) -> np.ndarray:
        """
        Resets the board of the env back to an initial chess board starting configuration for a new episode.
        """
        if self.record is True and self.board.move_stack:  # Record the moves of the prior game if non-empty
            output_dir = os.path.join(self.record_dir, f"game-{self.episode_id}")
            os.makedirs(output_dir, exist_ok=True)  # Create an output folder for this game recording
            save_recording(self.board.move_stack, os.path.join(output_dir, f"game-{self.episode_id}.mp4"))
            save_move_stack(self.board.move_stack, os.path.join(output_dir, f"game-{self.episode_id}.txt"))

        self.board = chess.Board()  # Start with the initial chess board configuration
        self.action_space = ActionSpace(self.board)  # Init an action space based on the current board
        self.episode_id += 1  # Incriment each time a new episode begins
        self.ep_ended, self.step_count = False, 0  # Reset the end flag and episode step counter
        return self.board.fen()  # Return the FEN representation which encodes the full state of the game

    def step(self, action: int) -> Tuple[Union[Dict, int, bool]]:
        """
        Takes an input action from the agent (encoded as an integer index of the action space) and runs 1
        timestep in the environment and returns a tuple of:
            (new_state, reward, is_terminated, is_truncated)

        :param action: An input action (int) to take from the current state.
        :return: A tuple of (new_state, reward, is_terminated, is_truncated).
        """
        assert self.ep_ended is not True, "Episode has already ended, step() cannot be called again"
        assert 0 <= action < self.action_space.n, "Action value outside of acceptable range"
        self.board.push(self.action_space.actions[action])  # Make the move specified in the env
        self.action_space = ActionSpace(self.board)  # Update the action space with the updated board config
        self.step_count += 1

        if self.board.is_game_over():  # Check if this results in the game ending by termination
            if self.board.is_checkmate():  # If the player's move results in a checkmate, the reward is +1
                reward = 1
            else:  # For stalemates and other types of draws, the reward is zero
                reward = 0
            terminated = True

        else:  # For all other intermediate states, the reward is zero
            reward = 0
            terminated = False

        truncated = (self.step_count >= self.step_limit)  # Check if the current ep reaches the step limit
        self.ep_ended = terminated or truncated
        # For the next state, return the FEN representation which encoded the full state of the game
        return self.board.fen(), reward, terminated, truncated

    def rev_step(self) -> Dict:
        """
        Reverses an action taken in the environment and returns the new starting state after the reversal.
        """
        assert self.step_count > 0, "Cannot reverse beyond initial start"
        self.board.pop()  # Reverse the last action taken
        self.action_space = ActionSpace(self.board)  # Update the action space with the updated board config
        self.step_count -= 1
        self.ep_ended = False  # If the last action ended the episode, then undoing it will reverse that
        return self.board.fen()

    def __repr__(self):
        """
        Displays the current board using the chess.Board __repr__ method.
        """
        return self.board.__repr__()

    def __str__(self):
        """
        Displays the current board using the chess.Board __str__ method.
        """
        return self.board.__str__()
