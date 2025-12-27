"""
This module configures an RL ChessAgent model along with a ChessEnv and runs a training loop to train the
model according to the config file specified by the --config command line argument.

The ChessEnv is based off the python-chess api, see https://python-chess.readthedocs.io/en/latest/ for
more details. A reward of +1 is given for checkmating the opponent, -1 for being checkmated, and a reward of
0 for all other game outcomes (e.g. stalemate or insufficient material). Training is done via self-play where
the RL agent plays both sides of the board simultaneously as black and white.

This module can be called from the command line e.g.
    > python run.py --config=MLP_Agent
or imported for running training in a Jupyter Notebook e.g.
    > from run import run_model_training
    > run_model_training("NatureDQN")
"""

import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CURRENT_DIR)

import argparse, shutil, torch
from utils.general import read_yaml
from utils.chess_env import ChessEnv
from core.base_components import LinearSchedule
from core.torch_models import ChessAgent

torch.set_float32_matmul_precision("high")  # Enables bf16 GEMMs on GPUs that support it

def run_model_training(config_name: str) -> None:
    """
    This function will run training for a specified config_name input e.g. MLP_Agent or CNN_Agent etc.

    :param config_name: The name of the config to use for model training.
    :return: None, results are saved to disk.
    """
    # 1). Read in the config file specified by the user to be used for model training
    config = read_yaml(os.path.join(CURRENT_DIR, f"config/{config_name}.yml"))

    for path_name, path in config["output"].items():  # Prepend the parent dir of this project to rel paths
        if path_name != "clear_all":  # The only config setting here that isn't a path
            config["output"][path_name] = os.path.join(CURRENT_DIR, path)

    if os.path.exists(config["output"]["output_path"]):  # Check if the output results directory exists
        if config["output"]["clear_all"] is True:  # Clear the existing contents if specified in the config
            shutil.rmtree(config["output"]["output_path"])  # Remove entire results directory

    # 2). Create the output directory for the results of this model if it doesn't already exist
    os.makedirs(config["output"]["output_path"], exist_ok=True)

    # 3). Instantiate the model to be trained by providing the env and config
    model = ChessAgent(config)

    # 4). Train the model or resume training from where it was last stopped
    model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the training, evaluation, and record loops for Chess RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",
                        help="The name of the config file in the config dir to be used for model training.")
    args = parser.parse_args()
    run_model_training(args.config)  # Run model training for the config file specified in the input arg
