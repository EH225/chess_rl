"""
This module contains general utility functions that are used throughout the training, evaluation, and
post-processing pipeline.
"""
import time, sys, logging, yaml, os, cv2
import numpy as np
from typing import Tuple, List
from collections import defaultdict

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


def process_recording(input_path: str, output_path: str, time_ds: int = 4, size_ds: int = 1,
                      max_len: int = 120, codec: str = 'mp4v') -> None:
    """
    Performs post-processing on a recording from the env. This function is able to down-sample temporally
    by retaining every Nth frame (controlled by time_ds), down-sample the size of the video frames by a
    factor of N (controlled by size_ds) and also cap the video length to a specified max_len in seconds.

    :param input_path: A file path to the original input video to be processed.
    :param output_path: A file paths for where the processed video should be written to.
    :param time_ds: A temporal down-sampling factor i.e. every Nth frame will be retained. If set to 1 then
        no temporal down-sampling is performed and the video recording remains that the original speed.
    :param size_ds: A frame size down-sampling factor i.e. if size_ds = 2 then the frame dimensions, height
        and width, will be reduced by a multiple of 2.
    :param max_len: A time limit it seconds for the output video, the default is 120 seconds = 2 minutes.
    :param codec: FourCC codec, default 'mp4v' which is usually best for handling mp4 files.
    :return: None, the output video is written to disk to output_path.
    """
    input_video = cv2.VideoCapture(input_path)  # Read in the input video from disk
    if not input_video.isOpened():
        raise RuntimeError(f"Cannot open video {input_path}")

    # Record the original video properties
    fps = input_video.get(cv2.CAP_PROP_FPS)  # Frames per second
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    # 4-byte code used to specify the video codec or compression format
    # fourcc = cv2.VideoWriter_fourcc(*codec)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Best option for HTML display compatibility

    # Perform frame dimension down-sampling if specified by size_ds
    new_width, new_height = width // size_ds, height // size_ds
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    frame_idx, frames_written = 0, 0
    total_frame_limit = max_len * fps  # The upper bound on how many frames to record

    while True:  # Write frames from the input video to the output video after processing them
        return_flag, frame = input_video.read()  # Read in the next frame from the input video
        if not return_flag or frames_written > total_frame_limit:
            break
        if frame_idx % time_ds == 0:  # Down-sample to every nth frame and re-size
            output_video.write(cv2.resize(frame, (new_width, new_height)))
            frames_written += 1
        frame_idx += 1

    input_video.release()
    output_video.release()


def video_post_processing(config: dict, time_ds: int = 4, size_ds: int = 1, max_len: int = 120) -> None:
    """
    Runs video post-processing on all .mp4 video files located in the record_path (if available) specified by
    the input config dictionary. This function performs post-processing recordings from the env created during
    training. This function is able to down-sample temporally by retaining every Nth frame (controlled by
    time_ds), down-sample the size of the video frames by a factor of N (controlled by size_ds) and also cap
    the video length to a specified max_len in seconds.

    :param config: An input config file specifying a config.output.record_path directory of .mp4 videos.
    :param time_ds: A temporal down-sampling factor i.e. every Nth frame will be retained. If set to 1 then
        no temporal down-sampling is performed and the video recording remains that the original speed.
    :param size_ds: A frame size down-sampling factor i.e. if size_ds = 2 then the frame dimensions, height
        and width, will be reduced by a multiple of 2.
    :param max_len: A time limit it seconds for the output video, the default is 120 seconds = 2 minutes.
    :return: None, the output videos are written to disk to to the record_path/postprocessed location.
    """
    if config["output"].get("record_path", None):  # If there is a record_path listed in the config
        output_path = os.path.join(config["output"]["record_path"], "postprocessed")
        os.makedirs(output_path, exist_ok=True)  # Make the save directory if not already there
        start_time = time.perf_counter()  # Report how long it takes to process the video recordings

        for filename in os.listdir(config["output"]["record_path"]):
            if filename.endswith(".mp4"):  # Operate only on the .mp4 files
                process_recording(input_path=os.path.join(config["output"]["record_path"], filename),
                                  output_path=os.path.join(output_path, filename),
                                  time_ds=time_ds, size_ds=size_ds, max_len=max_len)

        print(f"Video recording post processing complete! ({time.perf_counter() - start_time:.1f}s)")


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
    return logger


##############################
### Timer Class Definition ###
##############################
# TODO: Section marker

class Timer:
    """
    Timer object used to track the runtime for various operations segmented into various categories.
    """

    def __init__(self, enabled: bool = False) -> None:
        super().__init__()
        self.enabled = enabled
        # A bucket of [total_secs, latest_start, num_calls]
        self.category_sec_avg = defaultdict(lambda: [0.0, 0.0, 0])

    def start(self, category: str) -> None:
        """
        Method for recording the start of a given action instance from a specified category.
        """
        if self.enabled:  # Begin counting how long this action of type category takes
            stat = self.category_sec_avg[category]  # [total_secs, latest_start, num_calls]
            stat[1] = time.perf_counter()  # Record the latest_start time
            stat[2] += 1  # Incriment the number of times this operation has been called

    def end(self, category: str) -> None:
        """
        Method for recording the end of a given action instance from a specified category.
        """
        if self.enabled:  # End counting how long this action of type category takes
            stat = self.category_sec_avg[category]  # [total_secs, latest_start, num_calls]
            # Add how long the most recent execution took of an action of this type to the total_secs spend
            # on all actions of this type
            stat[0] += time.perf_counter() - stat[1]  # Now - start seconds = runtime duration

    def print_stat(self) -> None:
        """
        Prints a summary of runtimes for each action category tracked internally by this Timer object.
        """
        if self.enabled:  # Report the runtime stats for each type of operation if enabled
            print("Printing timer stats:")
            for key, val in self.category_sec_avg.items():  # Iter over all types of operations
                # val = [total_secs, latest_start, num_calls]
                if val[2] > 0:  # If we have called this operation at least 1x, then report it's avg runtime
                    print(f":> category {key}, total {val[0]}, calls {val[2]}, avg {val[0] / val[2]} s/call")

    def reset_stat(self) -> None:
        """
        Resets the internal state of the Timer object for all categories of actions i.e. clears all existing
        data on runtimes.
        """
        if self.enabled:
            print("Reseting all timer stats")
            for val in self.category_sec_avg.values():  # val = [total_secs, latest_start, num_calls]
                val[0], val[1], val[2] = 0.0, 0.0, 0


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
