"""
This module contains general utility functions that are used throughout the training, evaluation, and
post-processing pipeline.
"""
import time, sys, logging, yaml, os
import numpy as np
import pandas as pd
import socket
from typing import Tuple, List
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

logging.getLogger("matplotlib.font_manager").disabled = True

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

