"""
This module contains general utility functions that are used throughout the training, evaluation, and
post-processing pipeline.
"""
import time, sys, logging, yaml, os
import numpy as np
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


def save_eval_scores(eval_scores: List[Tuple[float]], save_dir: str) -> None:
    """
    Generates a time-series plot of the passed input eval_scores and saves the plot along with the data
    itself to CSV. eval_scores is expected to be a list of tuples (t, eval_score) reporting the eval scores
    of the model over various training iteration timestamps.

    :param eval_scores: An input list of evaluation score tuples from model training.
    :param save_dir: A directory in which to save the evaluation scores plot and data as a CSV.
    :return: None.
    """
    eval_scores = np.array(eval_scores)  # Convert from a list of tuples into a (N, 2) ndarray
    plt.figure(figsize=(8, 4))
    plt.plot(eval_scores[:, 0], eval_scores[:, 1], zorder=3)
    plt.xlabel("Training Timestep")
    plt.ylabel("Eval Score")
    plt.title("Evaluation Scores During Training")
    plt.grid(color="lightgray", zorder=-3)
    plt.savefig(os.path.join(save_dir, "eval_scores.png"))
    plt.close()

    # Write evaluation scores to a csv file
    np.savetxt(os.path.join(save_dir, "eval_scores.csv"), eval_scores, delimiter=", ", fmt="% s")


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
    return logger

################################
### Progbar Class Definition ###
################################
# TODO: Section marker

class Progbar:
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/).

    Displays a progress bar during training.
    """

    def __init__(self, target: int, width: int = 30, verbose: int = 1, discount: float = 0.9):
        """
        Initialize the progress bar training tracker.

        :param target: The total number of steps expected.
        :param width: The width of the progress bar displayed.
        :param verbose:  Controls the amount of reporting done by the progbar.
        :param discount:  Used to create an exponential moving average of recent values.
        """
        self.width = width  # Width of the progress bar
        self.target = target  # Number of total steps expected
        self.sum_values = {}
        self.exp_avg = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose
        self.discount = discount

    def reset_start(self):
        self.start = time.time()

    def update(self, current: int, values: list = None, exact: list = None, strict: list = None,
               exp_avg: list = None, base: int = 0) -> None:
        """
        This method updates the progress bar.

        :param current: The index of the current timestep.
        :param values: A list of tuples (name, value_for_last_step). The progress bar will display averages
            for these values.
        :param exact: A list of tuples (name, value_for_last_step). The progress bar will display these
            values directly.
        :param strict: A list of tuples (name, value_for_last_step)
        :param exp_avg: A list of tuples (name, value_for_last_step) used to update the EWAs.
        :param base: The starting number of iterations i.e. what current was at the beginning.
        :return: None.
        """
        values = values if values is not None else []
        exact = exact if exact is not None else []
        strict = strict if strict is not None else []
        exp_avg = exp_avg if exp_avg is not None else []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [(v * (current - self.seen_so_far), current - self.seen_so_far), ]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += current - self.seen_so_far

        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        for k, v in exp_avg:
            if k not in self.exp_avg:
                self.exp_avg[k] = v
            else:
                self.exp_avg[k] *= self.discount
                self.exp_avg[k] += (1 - self.discount) * v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = "%%%dd/%%%dd [" % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += "=" * (prog_width - 1)
                if current < self.target:
                    bar += ">"
                else:
                    bar += "="
            bar += "." * (self.width - prog_width)
            bar += "]"
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / (current - base)
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ""
            if current < self.target:
                info += " - ETA: %ds" % eta
            else:
                info += " - %ds" % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += " - %s: %.4f" % (
                        k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]),
                    )
                else:
                    info += " - %s: %s" % (k, self.sum_values[k])

            for k, v in self.exp_avg.items():
                info += " - %s: %.4f" % (k, v)

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += (prev_total_width - self.total_width) * " "

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = "%ds" % (now - self.start)
                for k in self.unique_values:
                    info += " - %s: %.4f" % (
                        k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]),
                    )
                sys.stdout.write(info + "\n")

    def add(self, n, values: list = None) -> None:
        values = values if values is not None else []
        self.update(self.seen_so_far + n, values)
