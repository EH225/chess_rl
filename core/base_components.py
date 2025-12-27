"""
This module contains the base components of training a deep value-network model. It defines a general class
for training DVN models.
"""
import logging
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import time, chess
import numpy as np
import pandas as pd
from tqdm import tqdm
from dask.distributed import Client, LocalCluster
from itertools import batched
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Callable, List, Tuple, Union, Dict, Optional

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from utils.general import get_logger, runtime, LinearSchedule, get_lan_ip
from utils.replay_buffer import ReplayBuffer
from utils.chess_env import ChessEnv, save_recording, save_move_stack, create_ep_record, move_stack_to_states
from utils.evaluate import evaluate_agent_game
import core.search_algos as search_algos


def setup_path(dask_worker):
    """
    This function is used to make sure all dask workers have the correct python path configured.
    """
    import sys, os
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)


#####################################
### Deep Value-Network Definition ###
#####################################
# TODO: Section marker

class DVN:
    """
    Base-class for implementing a Deep Value-Network RL model. Rather than computing Q-values over an action
    distribution, this model computes a value estimate of the current state under optimal play without
    explicitly mapping to a defined action space to avoid the high-dimensional complexity required for games
    such as chess.
    """

    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Initialize a Value Network and ChessEnv.

        :param env: A ChessEnv game environment used to get legal moves and board states.
        :param config: A config dictionary read from yaml that specifies hyperparameters.
        :param logger: A logger instance from the logging module for screen updates during training.
        :return: None.
        """
        # Configure the directory for training outputs
        os.makedirs(config["output"]["output_path"], exist_ok=True)

        # Store the hyperparams and other inputs
        self.config = config
        self.logger = logger if logger is not None else get_logger(config["output"]["log_path"])
        self.record_dir = config["output"]["record_path"]
        self.record = self.config["model_training"]["record"]

        # These are to be defined when self.initialize_model() is called
        self.v_network, self.optimizer = None, None

        # Auto-detect which device should be used by the model by what hardware is available
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        else:  # Default to using the CPU if no GPU accelerator
            self.device = "cpu"

        # Call the build method to instantiate the needed variables for the RL model
        self.build()

        # Configure a summary writer from TensorBoard for tracking the progress of training as we go
        self.summary_writer = SummaryWriter(self.config["output"]["tensorboard"], max_queue=int(1e5))

        # Configure the search function to be used for this model
        self._search_func = getattr(search_algos, config["search_func"]["name"])

    def initialize_model(self) -> None:
        """
        Initializes the required value network model.

        The input to this network will be a state representation of the current game state using FEN encoding
        and will be a string. self.v_network will be the learned value-function approximator. This method
        also instantiates an optimizer for training.
        """
        # This method is to be defined by an object in the torch_models module
        raise NotImplementedError

    def build(self) -> None:
        """
        Builds the model and performs necessary pre-processing steps
        1. Calls self.initialize_model() to instantiate the v_network model
        2. Loads in pre-trained weights if any are detected or randomly initializes them
        3. Moves the torch models to the appropriate device
        4. Compiles the model if specified in the config file
        """
        # 1). Initialize the v_network
        self.initialize_model()

        # 2). Load in existing pre-trained weights if they are available and specified by the config
        if "load_dir" in self.config["model_training"].keys():
            load_dir = self.config["model_training"]["load_dir"]
            self.logger.info(f"Looking for existing model weights and optimizer in: {load_dir}")
            if os.path.exists(load_dir):  # Check that the load weights / optimizer directory is valid

                # A). Attempt to load in pre-trained model weights from disk if they are available
                wts_path = os.path.join(load_dir, "model.bin")
                if os.path.exists(wts_path):  # Check if there is a cached model weights file
                    wts = torch.load(wts_path, map_location="cpu", weights_only=True)
                    self.v_network.load_state_dict(wts)  # Load in the model weights to the v_network
                    self.logger.info("Existing model weights loaded successfully!")

                # B). Attempt to load in an existing optimizer state if one is available
                opt_path = os.path.join(load_dir, "model.optim.bin")
                if os.path.exists(opt_path):  # Check if there is a cached model optimizer file
                    self.optimizer.load_state_dict(torch.load(opt_path, map_location="cpu",
                                                              weights_only=True))
                    self.logger.info("Existing optimizer weights loaded successfully!")
            else:
                self.logger.info("load_dir is not a valid directory")

        # NOTE: The code below is not strictly necessary, the weights are auto-initialized by PyTorch
        else:  # Otherwise if we're not using pre-trained weights, then initialize them randomly
            self.logger.info("Initializing parameters randomly")

            def init_random_weights(m):
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_uniform_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

            self.v_network.apply(init_random_weights)

        # 3). Now that we have the model and weights initialized, move them to the appropriate device
        self.v_network = self.v_network.to(self.device)
        self.v_network.device = self.device  # Update the device of the model after moving it

        trainable_params = sum(p.numel() for p in self.v_network.parameters() if p.requires_grad)
        self.logger.info(f"Total Trainable Parameters: {trainable_params}")

        for state in self.optimizer.state.values():  # Move the optimizer's state to the model's device
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # 4). Check if the model should be compiled or not, if so then attempt to do so
        if self.config["model_training"].get("compile", False):
            try:
                compile_mode = self.config["model_training"]["compile_mode"]
                self.v_network = torch.compile(self.v_network, mode=compile_mode)
                self.logger.info("Models compiled")
            except Exception as err:
                self.logger.info(f"Model compile attempted, but not supported: {err}")

    def get_best_action(self, state: str, default: int = None) -> Tuple[int, float, np.ndarray]:
        """
        This method is called by the train() and evaluate() methods to generate the best action at a given
        starting state, which is found by using the search function of this model and the current parameters
        of the v_network. This function returns the best action for the given input state, the overall
        estimate of the state's value, and also the value estimates for each possible action as well.

        :param state: A FEN (Forsyth–Edwards Notation) representation of the game state denoting the location
            of the pieces on the board, who is to move next, who can castle, en passant etc.
            E.g. 'r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4'
        :param default: A default action to take if state is None. Will return a randomly selected action
            if both state and default are None.
        :returns: A tuple of 3 elements:
            - The best action according to the model as an int
            - The value estimate of this state as a float
            - The estimated Q-values for all possible actions (best action is not always the argmax)
        """
        if state is None:  # If no state provided, then randomly sample from the action space and return a
            # collection of 0s for the estimated values of each action
            action = default if default is not None else self.env.action_space.sample()
            return action, 0.0, torch.zeros(self.env.action_space.n)

        with torch.no_grad():  # Gradient tracking is not needed for this online action-selection step
            # Use the search function to generate what the best action would be according to the model
            best_actions, state_values, action_values = self.search_func(state)

        return best_actions[0], state_values[0], action_values[0]

    @property
    def policy(self) -> Callable:
        """
        Returns a function that maps an input state (FEN game state) to the model's best action selection.
        """
        return lambda state: self.get_best_action(state)[0]

    def agent_move_func(self, board: chess.Board) -> chess.Move:
        """
        Given an input chess.Board object, this method returns the model's recommended best action given the
        current policy and greedy evaluation.

        :param board: An input chess.Board object representing the current game state.
        :return: A chess.Move that is the best action according to the learned RL model.
        """
        best_action, state_value, action_values = self.get_best_action(board.fen())
        return list(board.legal_moves)[best_action]

    def save(self):
        """
        Saves the parameters of self.v_network to the model_output directory specified in the config file
        along with the current state of the optimizer if any.
        """
        save_dir = self.config["output"]["model_output"]
        os.makedirs(save_dir, exist_ok=True)  # Make dir if needed
        if self.v_network is not None:  # If a v_network has been instantiated, save its weights
            torch.save(self.v_network.state_dict(), os.path.join(save_dir, "model.bin"))
        if self.optimizer is not None:  # If an optimizer has been instantiated, save its weights
            torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "model.optim.bin"))

    def search_func(self, state_batch: Union[str, List[str]]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        This method applies the search function specified in the config file to each state of the input state
        batch to compute a high-quality estimate of the value of each possible action.

        Returns the best_actions (ints), state_values (floats), action_values (np.ndarrays) where:
            - best_actions: A list of integers denoting the best action as per the search function
            - state_values: A list of floats denoting the overall value of the starting state
            - action_values: A list of torch.Tensor containing float estimates for the value of each possible
                action from the current state.

        :param state_batch: An input batch of states (a list of FEN strings) which encodes the game states
            from various starting positions.
        :return: Lists of best_actions (ints), state_values (floats), action_values (np.ndarrays)
        """
        if isinstance(state_batch, str):  # Accept a lone string, convert it to a list of size 1
            state_batch = [state_batch, ]  # All lines below expect state_batch to be a list

        best_actions = np.zeros(len(state_batch), dtype=np.uint16)  # 1 int best action value per state
        state_values = np.zeros(len(state_batch), dtype=np.float32)  # 1 float state estimate per state
        action_values = []  # Collect np.arrays in a list, each can be of variable length

        for i, state in enumerate(state_batch):
            best_action, state_value, action_vals = self._search_func(state=state, model=self.v_network,
                                                                      **self.config["search_func"])
            # Append the results from the search function evaluation of the state to the output lists
            best_actions[i] = best_action
            state_values[i] = state_value
            action_values.append(action_vals)

        return np.array(best_actions), np.array(state_values), action_values

    def compute_td_targets(self, state_batch: List[str], t: int) -> np.ndarray:
        """
        This method is used to compute the TD targets in parallel using dask which are used during training
        and are the most time consuming part of the training process. Each TD target calculation can be done
        in parallel since each computation is independent. This will greatly reduce the time it takes to
        generate the TD targets for gradient update steps.

        :param state_batch: An input batch of states (a list of FEN strings) which encodes the game states
            from various starting positions.
        :param t: The current training time step. This is used to control which search function and value
            estimator is used when generating the TD target values.
        :return: A np.ndarray of the same length as state_batch with an estimated state value from each
            derived from the v_network model and search function.
        """
        # 1). Save the current model weights so that they can be read from disk by each dask worker
        save_dir = self.config["output"]["model_output"]
        os.makedirs(save_dir, exist_ok=True)  # Make dir if needed
        if self.v_network is not None:  # If a v_network has been instantiated, save its weights
            torch.save(self.v_network.state_dict(), os.path.join(save_dir, "model.bin"))
        # Don't need to save the optimizer state here, we only will load in the model

        # 2). Split the input state_batch into equal parts according to the number of threads in the cluster
        nthreads = sum([worker.nthreads for worker in self.dask_client.cluster.workers.values()])
        state_batches = list(batched(state_batch, max(1, len(state_batch) // nthreads)))  # Equal parts

        # 3). Run the results in parallel using dask to speed up computations
        futures = [self.dask_client.submit(self._compute_td_targets, batch, self.config, t)
                   for batch in state_batches]
        results = []  # Consolidate the results in-order back into a list and then return a np.ndarray
        for res in self.dask_client.gather(futures):
            results.extend(list(res))
        return np.array(results)

    @staticmethod
    def _compute_td_targets(state_batch: List[str], config: Dict) -> np.ndarray:
        """
        This method sequentially computes the estimated value of each state in state_batch using the search
        function and model specified in the input config dictionary. A model instance is instantiated and the
        weights are read in from disk. Each state estimate is then computed using a search function and the
        results are collected and returned in a np.ndarray. This function is designed to be called in parallel
        using dask.

        :param config: A config dictionary read from yaml that specifies hyperparameters.
        :return: A np.ndarray of the same size as state_batch with state value estimates for each.
        """
        raise NotImplementedError

    def run_games(self, n_games: int, epsilon: float, t: int) -> List[str]:
        """
        Runs a series of on-policy n_games in parallel using dask to distribute the work load across all
        available threads. Actions are selected using an epsilon greedy strategy. This method returns a list
        of board states generated from the n_grames run.

        :param n_games: The number of on-policy games to run in parallel.
        :param epsilon: The exploration parameter i.e. with probability e the agent selects a random action.
        :return: A list of FEN strings encodings of all the board states reached during the n_games.
        """
        # 1). Save the current model weights so that they can be read from disk by each dask worker
        save_dir = self.config["output"]["model_output"]
        os.makedirs(save_dir, exist_ok=True)  # Make dir if needed
        if self.v_network is not None:  # If a v_network has been instantiated, save its weights
            torch.save(self.v_network.state_dict(), os.path.join(save_dir, "model.bin"))
        # Don't need to save the optimizer state here, we only will load in the model

        # 2). Run the games in parallel and aggregate the results from each
        # In order to get different results from each _run_game call, we have to differentiate the input so
        # we pass in i, the int counter which has no effect on the simulated games themselves
        futures = [self.dask_client.submit(self._run_game, epsilon, self.config, i) for i in range(n_games)]
        ep_records = []  # Collect the episode records from each game
        all_states = [] # Aggregate all the on-policy states generated from all n_games
        for (states, ep_record) in self.dask_client.gather(futures):
            ep_record["t"] = t
            ep_records.append(ep_record)
            all_states.extend(states)

        self.update_ep_history(ep_records, "train")  # Log all the training episodes in the csv log
        return all_states

    @staticmethod
    def _run_game(epsilon: float, config: Dict) -> Tuple[List[str], Dict]:
        """
        Runs 1 on-policy self-play chess match with an epsilon greedy action selection strategy. This function
        is designed to be called in parallel using dask to simulate many games simultaneously to generate
        state observations for the replay buffer during training.

        :param epsilon: The exploration parameter i.e. with probability e the agent selects a random action.
        :param config: A config dictionary read from yaml that specifies hyperparameters.
        :return: A list of game states (a list of FEN strings) and an ep_record summarizing the game.
        """
        raise NotImplementedError

    def calc_loss(self, v_est: torch.Tensor, v_search_est: torch.Tensor, wts: torch.Tensor
                  ) -> Tuple[torch.float, np.ndarray]:
        """
        Calculates the MSE loss of a batch of inputs. THe loss for an example is defined as:
            loss = (y - y_hat)^2 = (V_search(s) - V_hat(s))^2

        On the left, we use our search function derived estimate of this state's value which uses a lot of
        computation to make that determination and is a more reliable measure of true value than the y_hat,
        and this is the TD target value we wish to have our value approximator model learn to replicate.

        On the right, we use our self.v_network to estimate the value of the current state with only 1 forward
        pass through the model. This will be a less accurate estimate, but faster to compute and is also used
        in the application of the search function.

        This method returns the estimated loss across all samples in the batch and also the TD errors
        associated with each so that priority scores in the replay buffer can be updated.

        :param v_est: torch.Tensor with shape = (batch_size, )
            Estimated state values from the forward pass of the self.v_network i.e. the y_hats.
        :param v_search_est: torch.Tensor with shape = (batch_size, )
            Estimated state values derived from the running the search function on each i.e. the y values,
            or TD target values.
        :param wts: torch.Tensor with shape = (batch_size, )
            A weight vector for compute the MSE that is returned by the replay buffer sampling method to
            un-bias the gradient update using an importance sampling correction.
        :return:
            - A torch.float giving the MSE loss computed over all examples in the batch.
            - A np.ndarray denoting the |TD errors| for each example
        """
        assert len(v_est.shape) == 1, f"v_est expected to be 1 dimensional, got {v_est.shape}"
        msg = "v_search_est expected to be 1 dimensional, got {v_search_est.shape}"
        assert len(v_search_est.shape) == 1, msg
        td_errors = (v_est - v_search_est).abs()  # Compute the absolute value TD errors
        loss = (wts * td_errors.pow(2)).mean()  # Compute the weighted MSE loss function for all batch obs
        # After computing the loss, we should detach the td_errors from the gradient tracking computational
        # graph so that when we make updates to the replay buffer, gradients aren't being tracked there
        return loss, td_errors.detach().cpu().numpy()  # (torch.float, np.ndarray of size (batch_size, ))

    def train(self) -> None:
        """
        Runs a full training loop to train the parameters of self.v_network using the reply buffer.

        The epsilon-greedy exploration is gradually decayed over time and controlled by exp_schedule.
        The learning rate is gradually decayed over time and controlled by lr_schedule. The amount of
        importance sampling bias correction is controlled by beta_schedule.

        :param exp_schedule: A LinearSchedule instance where exp_schedule.param gives the internal epsilon
            parameter value.
        :param lr_schedule: A schedule for the learning rate where lr_schedule.param tracks it over time.
        :param beta_schedule: A schedule for the beta used in replay buffer weighted sampling.
        :return: None. Model weights and outputs are saved to disk periodically and also at the end.
        """
        global_start_time = time.time()
        self.logger.info(f"Training model: {self.config['model']}")
        self.logger.info(f"Running model training on device: {self.device}")

        local_cluster = LocalCluster(ip=get_lan_ip(), threads_per_worker=1, scheduler_port=8786)
        self.dask_client = Client(local_cluster) # Create a scheduler and connect it with the local cluster
        self.dask_client.register_worker_callbacks(setup_path)  # Configure sys.path of all workers
        # For any other computer on the network, activate the chess_rl venv and then run to add resources:
        #    dask-worker tcp://192.168.12.227:8786 --nthreads 11 --nworkers 16
        self.logger.info(f"Dask client scheduler address: {self.dask_client.scheduler.address}")

        # 0). Check that the v_network and optimizer are initialized
        for x in ["v_network", "optimizer"]:
            assert getattr(self, x) is not None, f"{x} is not initialized"

        # 1). Initialize the replay buffer and associated variables, it will keep track of recent obs so that
        #     we can maximize the amount of training we can get from them and set priority sampling
        replay_buffer = ReplayBuffer(size=self.config["hyper_params"]["buffer_size"],
                                     eps=self.config["hyper_params"]["eps"],
                                     alpha=self.config["hyper_params"]["alpha"],
                                     seed=self.config["model_training"].get("seed"))

        # 2). Initialize linear schedules for the learning rate, exploration rate and importance sampling
        #   A). Configure the exploration strategy with epsilon decay
        exp_schedule = LinearSchedule(float(self.config["hyper_params"]["eps_begin"]),
                                      float(self.config["hyper_params"]["eps_end"]),
                                      int(self.config["hyper_params"]["eps_nsteps"]))

        #   B) Configure the learning rate decay schedule
        lr_schedule = LinearSchedule(float(self.config["hyper_params"]["lr_begin"]),
                                     float(self.config["hyper_params"]["lr_end"]),
                                     int(self.config["hyper_params"]["lr_nsteps"]))

        #   C). Configure the beta importance sampling bias correction increase schedule
        beta_schedule = LinearSchedule(float(self.config["hyper_params"]["beta_begin"]),
                                       float(self.config["hyper_params"]["beta_end"]),
                                       int(self.config["hyper_params"]["beta_nsteps"]))

        # 3). Set up counter variables and resume training if cached results found on disk

        t = 0  # These counter vars are used by triggers for training, logging, evaluating etc.

        # If a train_ep_history.csv file exists on disk, then read it in and use that to update t
        ep_df_file_path = os.path.join(self.config["output"]["plot_output"], "train_ep_history.csv")
        if os.path.exists(ep_df_file_path):  # Check if a train_ep_history.csv file exists
            ep_df = pd.read_csv(ep_df_file_path)  # Read in the data from disk
            t = int(ep_df["t"].iloc[-1]) + 1  # Re-instate the largest t value recorded in the cached values

        # 4). Populate the replay buffer with some boards to start so that we have something to work with
        # before entering the main training loop
        exp_schedule.update(t)  # Update the epsilon obj before passing into the method below
        states = self.run_games(self.config["hyper_params"]["warm_up"], exp_schedule.param, t)
        replay_buffer.add_entries(states) # Add the on policy states generated during the eval game

        # 5). If we're starting from scratch, then run an eval episode to log the untrained performance
        if t == 0:
            score, states = self.evaluate(num_episodes=self.config["model_training"]["num_episodes_test"],
                                          record_last=self.record, return_states=True)
            replay_buffer.add_entries(states) # Add the on policy states generated during the eval game

        # 5). Run the full training loop, generate on-policy games, perform param updates, run evals
        n_steps_train = self.config["hyper_params"]["nsteps_train"]
        while t < n_steps_train:  # Loop until we reach the global training
            # Update the exploration rate, learning rate and beta as we go, update them for the current t
            exp_schedule.update(t)
            lr_schedule.update(t)
            beta_schedule.update(t)

            self.logger.info(f"\n[{t+1}/{n_steps_train}] ({(t+1)/n_steps_train:.2%}) " + "." * 100)
            iter_start = time.perf_counter() # Track how long the full training iteration takes
            # A). Play a series of on-policy games in parallel using dask to generate new states
            start_time = time.perf_counter() # Measure how long each step takes as well
            n_games = self.config["hyper_params"]["learning_freq"]
            states = self.run_games(n_games, exp_schedule.param, t)
            replay_buffer.add_entries(states) # Add the on policy states generated during the eval game
            msg = (f"\t{n_games} games and {len(states)} states generated with eps: {exp_schedule.param:.6f} "
                   f" ({runtime(start_time)})")
            self.logger.info(msg)

            # B). Perform training parameter update steps by sampling from the replay buffer
            for i in range(self.config["hyper_params"]["learning_updates"]):
                self.logger.info("\tStarting gradient update step...")
                start_time = time.perf_counter() # Measure how long each step takes
                lr, beta = lr_schedule.param, beta_schedule.param
                loss, grad_norm = self.update_step(replay_buffer, lr, beta, t)
                msg = (f"\tGradient update complete with lr: {lr:.6f}, beta: {beta:.4f}, loss: {loss:.4f}, "
                       f"grad_norm: {grad_norm:.4f} ({runtime(start_time)})")
                self.logger.info(msg)

            # C). Periodically save the model weights and optimizer state during training
            if t > 0 and t % self.config["model_training"]["saving_freq"] == 0:
                start_time = time.perf_counter() # Measure how long each step takes
                self.save()
                self.logger.info(f"\tModel saved ({runtime(start_time)})")

            if t == self.config["pre_train"]["nsteps_pretrain"] - 1:
                # If this is the last iteration of the pre-training, then save the model weights to a
                # separate location so that we have an archive of them to compare vs the final weights
                start_time = time.perf_counter() # Measure how long each step takes
                save_dir = os.path.join(self.config["output"]["model_output"], "pretraining")
                os.makedirs(save_dir, exist_ok=True)  # Make dir if needed
                torch.save(self.v_network.state_dict(), os.path.join(save_dir, "pretrain_model.bin"))
                self.logger.info(f"\tPretrained model weights saved ({runtime(start_time)})")

            # D). Periodically run an evaluation episode
            if t > 0 and t % self.config["model_training"]["eval_freq"] == 0:
                start_time = time.perf_counter() # Measure how long each step takes
                n_ep = self.config["model_training"]["num_episodes_test"]
                score, states = self.evaluate(num_episodes=n_ep, record_last=self.record,
                                              return_states=True)  # Compute a new eval score value
                replay_buffer.add_entries(states) # Add the on policy states generated during the eval game
                msg = f"\t{n_ep} eval episode(s) run with score: {score:.1f} ({runtime(start_time)})"
                self.logger.info(msg)

            self.logger.info(f"Full training iteration complete ({runtime(iter_start)})")
            t += 1  # Increment the global training step counter after completing another training iteration

        # Final screen updates
        self.logger.info("Training Complete!")
        self.save()  # Save the final model weights after training has finished
        self.dask_client.shutdown()  # Shutdown the dask cluster after training has finished
        # Run another evaluation episode at the very end, this will auto log the results to the CSV history
        score, states = self.evaluate(num_episodes=self.config["model_training"]["num_episodes_test"],
                                      record_last=self.record, return_states=True)
        duration = time.time() - global_start_time
        self.logger.info(f"Training finished in {duration / 60:.1f} minutes")

    def update_step(self, replay_buffer: ReplayBuffer, lr: float, beta: float, t: int) -> Tuple[int, int]:
        """
        Performs an update of the self.v_network parameters by sampling from replay_buffer.

        :param replay_buffer: A ReplayBuffer instance where .sample() gives us batches.
        :param lr: The learning rate to use when making gradient descent updates to self.v_network.
        :param beta: A hyperparameter used in prioritized experience replay sampling.
        :param t: The current training time step. This is used to control which search function and value
            estimator is used when generating the TD target values.
        :return: The loss = (y - y_hat)^2 and the total norm of the parameter gradients.
        """
        batch_size = self.config["hyper_params"]["batch_size"]

        # 1). Sample from the reply buffer to get recent states (encoded a FEN strings)
        start_time = time.perf_counter() # Measure how long each step takes
        state_batch, wts, indices = replay_buffer.sample(batch_size, beta)
        self.logger.info(f"\t   Sampled from replay buffer ({runtime(start_time)})")

        # [state_batch, wts, indices] -> (batch_size, ) lists of values

        # 2). Check that required components are present
        assert self.v_network is not None, "WARNING: Networks not initialized. Check initialize_model"
        assert self.optimizer is not None, "WARNING: Optimizer not initialized. Check add_optimizer"

        # 3). Zero the tracked gradients of the v_network model
        self.optimizer.zero_grad()

        # 4). Run a forward pass of the batch of states through the v_network and generate value estimates
        # i.e. what the v_network estimates is the discounted future return of following an optimal policy
        # from each state as the initial starting state
        start_time = time.perf_counter() # Measure how long each step takes
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):  # Use BFloat16
            v_est = self.v_network(state_batch).reshape(-1)  # Returns a torch.Tensor on self.device
        self.logger.info(f"\t   {len(v_est)} y-hat state value estimates generated ({runtime(start_time)})")
        v_est_np = v_est.detach().to(torch.float32).cpu().numpy() # Convert over to numpy for reporting
        summary_stats = [f"{x:.2f}" for x in [v_est_np.max(), v_est_np.min(), v_est_np.mean(),
                                              np.abs(v_est_np).mean(), v_est_np.std()]]
        summary_stats = "(max, min, mean, |mean|, std) = (" + ", ".join(summary_stats) + ")"
        self.logger.info(f"\t   v_est: {summary_stats}")
        v_est_np_model = v_est_np  # Save the alias

        # 5). Compute the TD target values using a search function to get more accurate values by looking
        # ahead, we want to train the network to estimate this search function in the forward pass
        start_time = time.perf_counter() # Measure how long each step takes
        v_search_estimates = self.compute_td_targets(state_batch, t)
        msg = f"\t   {len(v_search_estimates)} y state value estimates generated ({runtime(start_time)})"
        self.logger.info(msg)
        v_est_np = v_search_estimates.copy()
        summary_stats = [f"{x:.2f}" for x in [v_est_np.max(), v_est_np.min(), v_est_np.mean(),
                                              np.abs(v_est_np).mean(), v_est_np.std()]]
        summary_stats = "(max, min, mean, |mean|, std) = (" + ", ".join(summary_stats) + ")"
        corr = np.corrcoef(v_est_np_model, v_est_np)[0, 1] # Compute the correlation of the ys and yhats
        self.logger.info(f"\t   v_est: {summary_stats}, corr: {corr:.2f}")

        # 6). Compute gradients wrt to the MSE Loss function
        # Convert the inputs for this calculation to torch.Tensor and move to self.device
        wts = torch.from_numpy(wts).to(self.device)
        v_search_estimates = torch.from_numpy(v_search_estimates).to(self.device)

        start_time = time.perf_counter() # Measure how long each step takes
        loss, td_errors = self.calc_loss(v_est, v_search_estimates, wts)  # td_errors is a np.array
        replay_buffer.update_priorities(indices, td_errors)  # Update the priorities for the obs sampled
        loss.backward()  # Compute gradients wrt to the trainable parameters of self.v_network

        # total_norm records the L2 norm across all gradients i.e. the pre-clipping global gradient norm
        # which will then be scaled down to match clip_val if larger, otherwise it remains unchanged
        total_norm = torch.nn.utils.clip_grad_norm_(self.v_network.parameters(),
                                                    self.config["model_training"]["clip_val"])

        # 7). Update parameters with the optimizer by taking a step
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()  # Perform a gradient descent update step for all parameters
        msg = f"\t   Gradients computed and parameters updated ({runtime(start_time)})"
        self.logger.info(msg)

        return loss.item(), total_norm.item()  # Return the loss and the norm of the gradients

    def evaluate(self, num_episodes: int = None, record_last: bool = True, return_states: bool = False,
                 verbose: bool = True) -> float:
        """
        Runs a series of self-pay games (num_episodes) using the current model parameters with greedy action
        selection to evaluate the current parameter set and returns and average per episode score.

        Score here is defined using ACPL = Average Centipawn Loss which uses stockfish to evaluate the board
        before and after a move is make and computes the difference to create a quality loss metric. The lower
        the quality loss, the better. This metric allows us to evaluate sel-play matches, is highly correlated
        to  Elo score and is much lower variance which means fewer games are needed to establish a score.

        :param num_episodes: The number of episodes to run to compute an average per episode score.
        :param record_last: If set to True, then the last game will be recorded as a video save saved to disk.
        :param verbose: Indicates whether the results of the evaluation run should be reported via logging.
        :return: The average per episode socre over num_episodes.
        """
        self.logger.info(f"\nRunning Evaluation for {num_episodes} episode(s)")
        max_moves = self.config["model_training"]["max_eval_moves"]
        all_losses = []
        states = []  # If return_states is True, then also record the states generated during play

        with logging_redirect_tqdm():
            for g in tqdm(range(num_episodes), ncols=75):
                # Compute the losses for each move for 1 self-play game
                ep_losses, move_stack = evaluate_agent_game(self.agent_move_func, max_moves=max_moves,
                                                            return_move_stack=True)
                all_losses.extend(ep_losses)  # Aggregate the losses over all moves and all games
                self.update_ep_history([create_ep_record(move_stack)], prefix="eval")  # Record game summary
                if verbose is True:
                    msg = f"Game {g + 1}: ACPL = {sum(ep_losses) / len(ep_losses):.1f}, {len(move_stack)}"
                    msg += " moves total"
                    self.logger.info(msg)

                if return_states is True:
                    states.extend(move_stack_to_states(move_stack))

        overall_acpl = sum(all_losses) / len(all_losses)
        if verbose is True:
            self.logger.info(f"\nOverall ACPL: {overall_acpl:.1f}")

        if record_last is True:  # Save the last evaluation episode as a recording
            episode_id = 0 # Default to 0 to start with
            # Ideally we want to keep the episode_id recording numbers in sync with the eval history episodes
            ep_df_file_path = os.path.join(self.config["output"]["plot_output"], "eval_ep_history.csv")
            if os.path.exists(ep_df_file_path):  # Check if a eval_ep_history.csv file exists
                ep_df = pd.read_csv(ep_df_file_path)  # Read in the data from disk
                episode_id = int(ep_df["episode_id"].iloc[-1]) + 1  # Update from the cached log file

            output_dir = os.path.join(self.record_dir, f"eval-game-{episode_id}")
            os.makedirs(output_dir, exist_ok=True)  # Create an output folder for this game recording
            save_recording(move_stack, os.path.join(output_dir, f"eval-game-{episode_id}.mp4"))
            save_move_stack(move_stack, os.path.join(output_dir, f"eval-game-{episode_id}.txt"))

        return (overall_acpl, states) if return_states else overall_acpl

    def update_ep_history(self, ep_records: List[Dict], prefix: str = "train") -> None:
        """
        Appends the info of the input ep_records to an existing {prefix}_ep_history.csv if one exists,
        otherwise this method creates a new csv file and saves down the episode record summary as the first
        result. This method is generally called after running a full training or evaluation episode and is
        used to add episode summary entries to the output cache.

        :param ep_records: A list of episode records to record as a list of dictionaries.
        :param prefix: A prefix value for the {prefix}_ep_history.csv file.
        :return: None, updates the local {prefix}_ep_history.csv cache or creates one.
        """
        filename = "ep_history.csv" if not prefix else f"{prefix}_ep_history.csv"
        ep_df_file_path = os.path.join(self.config["output"]["plot_output"], filename)
        if os.path.exists(ep_df_file_path):  # Check if an {prefix}_ep_history file has been cached
            ep_df = pd.read_csv(ep_df_file_path)  # Read in the data from disk
        else:
            ep_df = None

        for ep_record in ep_records:
            ep_record = pd.Series(ep_record)
            if ep_df is not None:
                ep_record["episode_id"] = len(ep_df)
                ep_df.loc[len(ep_df), :] = ep_record
            else:  # If not, then use this ep_record as the first entry in the ep_df
                ep_df = ep_record.to_frame().T

        for col in ["outcome", "winner", "end_state"]:  # Change these columns to string format
            ep_df[col] = ep_df[col].astype(str)

        ep_df.to_csv(ep_df_file_path, index=False)  # Write out a new copy of the ep_df to disk to cache

    def record(self, max_moves: int = 250, episode_id: Optional[int] = None) -> None:
        """
        This method records a video for 1 episode using the model's current weights and saves it to disk.
        Videos are saved to self.config["output"]["record_path"] and are named: game-{ep} where ep is the
        episode counter from the environment which will monotonically increase during training.

        :param max_moves: An integer denoting the max number of moves allowed per game.
        :param episode_id: Used to name the output video i.e. game-{episode_id}. If not specified, then the
            first positive int above the prior file names found will be used.
        :return: None. Records a video of a self-play game and saves it to disk.
        """
        env = ChessEnv(step_limit=max_moves, record_dir=self.config["output"]["record_path"])
        if episode_id is None: # If not specified, then attempt to infer from the existing directory
            episode_id = 0 # Default to 0
            for file_name in os.listdir(self.config["output"]["record_path"]):
                if os.path.isdir(file_name): # If it's a directory, try to parse
                    try: # Expected directory names: "game-{int}"
                        episode_id = max(episode_id, int(file_name.split("-")[-1]) + 1)
                    except:
                        pass
        env.episode_id = episode_id  # Set for recording
        state = env.reset()
        env.record = True  # Toggle on the recording setting within the ChessEnv
        policy = self.policy() # Create a policy func that return the best move index according to the model

        terminated, truncated = False, False
        while not (terminated or truncated):
            new_state, reward, terminated, truncated = env.step(policy(state))
            state = new_state  # Update for next iteration

        state = env.reset()  # The env reset triggers the steps in the move_stack to be recorded
