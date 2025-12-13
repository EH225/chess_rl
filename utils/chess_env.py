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


def board_svg_to_arr(board: chess.Board) -> np.ndarray:
    """
    This function takes in a chess.Board object and converts the svg image representation to a np.ndarray
    of pixel values for writing frames to an mp4 video files.

    :param board: The input chess.Board from which to extract the svg image of the current board.
    :return: A np.ndarray of the current board including highlights and board annotations.
    """
    svg = board._repr_svg_()  # Extract the current board svg with annovations and overlays
    png_bytes = cairosvg.svg2png(bytestring=svg.encode())
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return np.array(img)


def save_recording(move_stack: List[chess.Move], output_path: str, fps: int = 1) -> None:
    """
    Saves a recording of a game which evolves through time according to move_stack, a python list of moves
    that can be executed in a chess.Board env.

    :param move_stack: A list of chess.Move objects describing the evolution of the game.
    :param output_dir: The output path to write the recording to ending in .mp4.
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
        print("Exception occured", e)

    out.release()  # Make sure to release even if there is an error
    print(f"Recording written to: {output_path}")


def save_move_stack(move_stack: List[chess.Move], output_path: str) -> None:
    """
    Saves a sequence of moves contained in move_stack to a text file.

    :param move_stack: A list of chess.Move objects describing the evolution of the game.
    :param output_dir: The output path to write the recording to ending in .txt.
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

    def __init__(self, step_limit: int = 5000, record_dir: str = None, initial_state: str = None):
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

        self.state = chess.Board()  # Start with the initial chess board configuration
        self.action_space = ActionSpace(self.state)  # Init an action space based on the current board
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

        truncated = (self.step_count == self.step_limit)  # Check if the current ep reaches the step limit
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
