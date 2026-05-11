"""
This module contains general utility functions that are used throughout the training, evaluation, and
post-processing pipeline.
"""
import time, sys, logging, yaml, os
import numpy as np
import pandas as pd
import socket, torch
from typing import Tuple, List, Dict
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

logging.getLogger("matplotlib.font_manager").disabled = True


def create_move_to_idx_map() -> Dict:
    """
    Creates a mapping between UCI moves e.g. "a1a2" or "f6f7q" denoting a to-from grid cell pairing
    with a unique integer value. There are 1968 in total.

    :returns: A length 1968 dictionary of all possible UCI moves and their associated integer indices starting
        at 0 and going to 1967.
    """
    files, ranks = 'abcdefgh', '12345678'
    uci_moves = []

    # Queen (q_dirs) directions covers the movement patterns of all pieces on the board except knights
    q_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    # Knight (k_dirs) gives us the other possible movement patterns
    k_dirs = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]
    promos = ['n', 'b', 'r', 'q']  # All the possible pieces that a pawn can be promoted into

    for r in range(8):  # Iterate over all ranks (rows) on the board
        for f in range(8):  # Iterate over all files (cols) on the board
            start = files[f] + ranks[r]  # Create the starting position e.g. a1

            # 1. Consider moves in the Queen planes (56)
            for df, dr in q_dirs:
                for dist in range(1, 8):
                    f2, r2 = f + df * dist, r + dr * dist
                    if 0 <= f2 < 8 and 0 <= r2 < 8:
                        move = start + files[f2] + ranks[r2]  # e.g. a1a2
                        uci_moves.append(move)

            # 2. Consider moves in the Knight planes (8)
            for df, dr in k_dirs:
                f2, r2 = f + df, r + dr
                if 0 <= f2 < 8 and 0 <= r2 < 8:
                    uci_moves.append(start + files[f2] + ranks[r2])

            # 3. Consider promotion moves as well
            for r_start, r_end in [(6, 7), (1, 0)]:  # Can only promote on the first or last row
                if r == r_start:
                    for df in [-1, 0, 1]:  # Promotion can be by vertical movement or diagonal capture
                        f2 = f + df
                        if 0 <= f2 < 8:
                            for p in promos:
                                uci_moves.append(start + files[f2] + ranks[r_end] + p)

    return {m: i for i, m in enumerate(uci_moves)}


def compute_img_out_dim(input_dims: Tuple[int], kernel_size: int, padding: int = 0, dilation: int = 1,
                        stride: int = 1) -> Tuple[int]:
    """
    Computes the output dimensions (h, w) of each image after being passed through a nn.Conv2d layer.

    Each image comes in with dimensions input_dims (h, w) and after the convolutions are run, the image
    shape may be altered based on the kernel_size, padding, stride, and dilation.
    """
    h, w = input_dims  # Unpack the original input dims provided
    h_out = (h + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    w_out = (w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return h_out, w_out


def join_path(loader, node):
    """
    Define a helper method to apply within the config yaml files to join together file paths.
    """
    return os.path.join(*[str(x) for x in loader.construct_sequence(node)])


yaml.add_constructor("!join_path", join_path)  # Needed so that yaml can process the join_path commands


def read_yaml(file_path: str) -> dict:
    """
    Helper function that reads in a yaml file specified and returns the associated data as a dict.

    :param file_path: A str denoting the location of a yaml file to read in.
    :return: A dictionary of data read in from the yaml file located at file_path.
    """
    return yaml.load(open(file_path), Loader=yaml.FullLoader)


def get_logger(log_filename: str) -> logging.Logger:
    """
    Returns a logging.Logger instance that will write log outputs to a filepath specified.
    """
    logger = logging.getLogger("logger")  # Init a logger
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    handler = logging.FileHandler(log_filename)  # Configure the logging output file path
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger("chess.engine").setLevel(logging.INFO)  # Supress printouts from the chess env
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)
    logging.getLogger("distributed.utils").setLevel(logging.ERROR)
    return logger


def runtime(start_time: float) -> str:
    """
    Given an input start_time = time.perf_counter(), this function returns the total runtime in string
    format e.g. 2.3s using the current time of when this function is called.

    :param start_time: The start time as a float for when the interval began.
    :return: A string denoting the runtime of the time interval in seconds e.g. 2.3s ending the interval now.
    """
    return f"{time.perf_counter() - start_time:.1f}s"


def convert_seconds(seconds: float) -> str:
    """
    Converts an input number of total seconds into a [x]h [y]m [z]s time string.

    :param seconds: The number of total seconds as an input.
    :return: The total seconds input converted to [x]h [y]m [z]s format.
    """
    hours = seconds // 3600  # Get the number of hours
    minutes = (seconds % 3600) // 60  # Get the number of minutes
    seconds = seconds % 60  # Get the remaining seconds
    return f"{int(hours)}h {int(minutes)}m {int(round(seconds, 0))}s"


class LinearSchedule:
    """
    Sets a linear schedule for the linear evolution of a given parameter over time e.g. learning rate,
    epsilon exploration rate, sampling beta.
    """

    def __init__(self, param_begin: float, param_end: float, nsteps: int):
        """
        Initializes a LinearSchedule object instance with an update() method that will update a parameter
        (e.g. a learning rate or epsilon exploration rate) being tracked at self.param.

        :param param_begin: The exploration parameter epsilon's starting value.
        :param param_end: The exploration parameter epsilon's ending value.
        :param nsteps: The number of steps over which the exploration parameter epsilon will decay from
            eps_begin to eps_end linearly.
        :returns: None
        """
        self.param = param_begin  # epsilon beings at eps_begin
        self.param_begin = param_begin
        self.param_end = param_end
        self.nsteps = nsteps
        # Using a linear decay schedule, the amount of decay for each timestep will be equal each time so
        # we can pre-compute the size of each decay step and store that here
        self.update_per_step = ((self.param_end - self.param_begin) / self.nsteps)

    def update(self, t: int) -> float:
        """
        Updates param internally at self.param using a linear interpolation from self.param _begin to
        self.param_end as t goes from 0 to self.nsteps. For t > self.nsteps self.param remains constant as
        the last updated self.param value, which is self.param_end.

        :param t: The time index i.e. frame number of the current step.
        :return: The updated exploration parameter value.
        """
        if t < self.nsteps:  # Prior to the end of the decay schedule, compute the linear decay +1 step
            self.param = self.param_begin + self.update_per_step * t
        else:  # After nsteps, set param to param_end
            self.param = self.param_end
        return self.param


def get_device() -> str:
    """
    Auto detects what hardware is available and returns a device name accordingly.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:  # Default to using the CPU if no GPU accelerator
        return "cpu"


def get_lan_ip():
    """
    Returns the LAN IP of the local network.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable — just forces the OS to pick the LAN interface
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def get_amp_dtype(device: str = "cuda"):
    """
    Determines the Automatic Mixed Precision data type that can be used on the current hardware.

    :param device: The device currently available as a string e.g. "cpu" or "cuda".
    :returns: A torch float type for auto mixed precision training.
    """
    assert isinstance(device, str), "device must be a str"
    if device != "cuda" or not torch.cuda.is_available():
        return torch.float16

    # Get compute capability (major, minor)
    major, minor = torch.cuda.get_device_capability()

    # Ampere (8.x), Hopper (9.x), Ada (8.9) → BF16 supported
    bf16_supported = (major >= 8)

    return torch.bfloat16 if bf16_supported else torch.float16
