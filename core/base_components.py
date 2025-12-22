"""
This module contains the base components of training a deep value-network model. It defines linear schedule
objects for decaying the learning rate and epsilon exploration parameter (epsilon) over time and a general
class for training DVN models.
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
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Callable, List, Tuple, Union, Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from utils.general import get_logger, Progbar
from utils.replay_buffer import ReplayBuffer
from utils.chess_env import ChessEnv, save_recording, save_move_stack, material_diff, create_ep_record
from utils.general import save_eval_scores
from utils.evaluate import evaluate_agent_game
import core.search_algos as search_algos

#########################################
### Exploration Rate Decay Schedulers ###
#########################################
# TODO: Section marker

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

    def __init__(self, env: ChessEnv, config: dict, logger: logging.Logger = None):
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
        self.env = env
        self.config = config
        self.logger = logger if logger is not None else get_logger(config["output"]["log_path"])
        self.record_dir = config["output"]["record_path"]

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

            def init_weights_randomly(m):
                if hasattr(m, "weight"):
                    torch.nn.init.xavier_uniform_(m.weight, gain=2 ** (1.0 / 2))
                if hasattr(m, "bias"):
                    torch.nn.init.zeros_(m.bias)

            self.v_network.apply(init_weights_randomly)

        # 3). Now that we have the model and weights initialized, move them to the appropriate device
        self.v_network = self.v_network.to(self.device)

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
        if state is None:  # If not state provided, then randomly sample from the action space and return a
            # collection of 0s for the estimated values of each action
            action = default if default is None else self.env.action_space.sample()
            return action, torch.zeros(self.env.action_space.n)

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

    def init_averages(self):
        """
        Defines extra attributes for monitoring the training with tensorboard.
        """
        self.avg_reward, self.max_reward, self.eval_reward = 0.0, 0.0, 0.0
        self.avg_q, self.max_q, self.std_q, self.std_reward = 0.0, 0.0, 0.0, 0.0

    def update_averages(self, rewards: deque, max_q_values: deque, q_values: deque,
                        scores_eval: list) -> None:
        """
        Updates the rewards averages and other summary stats for tensorboard.

        :param rewards: A deque of recent reward values.
        :param max_q_values: A deque of max q-values.
        :param q_values: A deque of recent q_values.
        :param scores_eval: A list of recent evaluation scores.
        :return: None.
        """
        if len(rewards) > 0:
            self.avg_reward = np.mean(rewards)  # Record the mean of recent rewards
            self.max_reward = np.max(rewards)  # Record the max of recent rewards
            self.std_reward = np.std(rewards)  # Record the std of recent rewards

        self.avg_q = np.mean(q_values)  # Record the mean of recent q-values
        self.max_q = np.mean(max_q_values)  # Record the mean of recent max q-values
        self.std_q = np.std(q_values)  # Record the std of recent rewards

        if len(scores_eval) > 0:  # If we have computed at least 1 evaluation score
            self.eval_reward = scores_eval[-1][1]  # Record the most recent evaluation score

    def add_summary(self, latest_loss: float, latest_total_norm: float, t: int) -> None:
        """
        Configurations for Tensorboard performance tracking.
        """
        self.summary_writer.add_scalar("loss", latest_loss, t)
        self.summary_writer.add_scalar("grad_norm", latest_total_norm, t)
        self.summary_writer.add_scalar("Avg_Reward", self.avg_reward, t)
        self.summary_writer.add_scalar("Max_Reward", self.max_reward, t)
        self.summary_writer.add_scalar("Std_Reward", self.std_reward, t)
        self.summary_writer.add_scalar("Avg_Q", self.avg_q, t)
        self.summary_writer.add_scalar("Max_Q", self.max_q, t)
        self.summary_writer.add_scalar("Std_Q", self.std_q, t)
        self.summary_writer.add_scalar("Eval_Reward", self.eval_reward, t)

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

    def _populate_buffer(self, replay_buffer: ReplayBuffer, exp_schedule: LinearSchedule,
                         episode_rewards: deque, max_q_values: deque, q_values: deque, n_iters: int) -> None:
        """
        Helper method for populating the replay buffer and other training performance tracking variables
        when model training is stopped and re-started again. The replay buffer is not cached on disk, so when
        we resume training, it helps to populate the replay buffer with values before sampling from it.

        :param replay_buffer: A ReplayBuffer object which will have n_iters frames added to it.
        :param exp_schedule: A LinearSchedule instance where exp_schedule.param gives the internal epsilon
            parameter value.
        :param episode_rewards: A deque tracking recent full-episode rewards.
        :param max_q_values: A deque tracking recent max q-values.
        :param q_values: A deque tracking recent average q-values.
        :param n_iters: The number of warm-up training step iterations to use to populate the training vars.
        :return: None, modifies the passed objects in place.
        """
        t = 0
        while t <= n_iters:  # Run iterations in the env until we reach the n_iters limit
            episode_reward = 0  # Track the total reward from all actions during the episode
            state = self.env.reset()  # Reset the env to begin a new training episode
            replay_buffer.add_entry(state)  # Add the starting board to the replay buffer

            while True:  # Run an episode of obs -> action -> obs -> action in the env until finished which
                # happens when either 1). the episode has been terminated by the env 2). the episode has
                # been truncated by the env or 3). the total training steps taken exceeds nsteps_train
                t += 1  # Track how many frames have been added to the replay buffer so far

                sys.stdout.flush()  # Screen updates while populating the replay buffer
                sys.stdout.write(f"\rPopulating the replay buffer {t}/{n_iters}...")

                # Choose and action according to current V Network and exploration parameter epsilon
                if np.random.rand() < exp_schedule.param:  # With probability epsilon, choose a random action
                    action, state_value, action_values = self.env.action_space.sample(), 0, np.zeros(0)
                else:  # Otherwise actually use the model to evaluate
                    action, state_value, action_values = self.get_best_action(state)

                # Store the q values from the learned v_network in the deque data structures
                if len(action_values) > 0:
                    max_q_values.append(np.max(action_values))  # Keep track of the max q-value
                    q_values.append(np.mean(action_values))  # Keep track of the avg q-value

                # Perform the selected action in the env, get the new state, reward, and stopping flags
                new_state, reward, terminated, truncated = self.env.step(action)
                reward = np.clip(reward, -1, 1)  # We expect +/-1, but add reward clipping for safety

                # Record the new state after taking this action in the replay buffer
                replay_buffer.add_entry(new_state)

                # Track the total reward throughout the full episode
                episode_reward += reward

                state = new_state  # Update for next iteration

                # End the episode if one of the stopping conditions is met
                if terminated or truncated or t >= n_iters:
                    break

            episode_rewards.append(episode_reward)  # Record the total reward received during the last episode

    def train(self, exp_schedule: LinearSchedule, lr_schedule: LinearSchedule,
              beta_schedule: LinearSchedule) -> None:
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
        start_time = time.time()
        self.logger.info(f"Training model: {self.config['model']}")
        self.logger.info(f"Running model training on device: {self.device}")

        # 0). Check that the v_network and optimizer are initialized
        for x in ["v_network", "optimizer"]:
            assert getattr(self, x) is not None, f"{x} is not initialized"

        # 1). Initialize the replay buffer and associated variables, it will keep track of recent obs so that
        #     we can maximize the amount of training we can get from them and set priority sampling
        replay_buffer = ReplayBuffer(size=self.config["hyper_params"]["buffer_size"],
                                     eps=self.config["hyper_params"]["eps"],
                                     alpha=self.config["hyper_params"]["alpha"],
                                     seed=self.config["model_training"].get("seed"))

        # 2). Collect recent rewards and q-values in deque data structures and init other tracking vars
        # Track the rewards after running each episode to completion or truncation / termination
        episode_rewards = deque(maxlen=self.config["model_training"]["num_episodes_test"])
        max_q_values = deque(maxlen=1000)  # Track the recent max_q_values we get from the v_network search
        q_values = deque(maxlen=1000)  # Track the recent q_values from the v_network across all actions
        self.init_averages()  # Used for tracking progress via Tensorboard

        t, last_eval = 0, 0  # These counter vars are used by triggers
        # t = tracks the global number of timesteps so far i.e. how many time we call self.env.step(action)
        # last_eval = records the value of t at which we last ran an self.evaluation()

        # First look if there is an eval_score.csv file already on disk, if so, read it in and continue adding
        # to it from there, use it to infer the t that we will continue training at
        file_path = os.path.join(self.config["output"]["plot_output"], "eval_scores.csv")
        if os.path.exists(file_path):  # Check if an eval score file has been cached to the output directory
            eval_scores = [tuple(x) for x in np.loadtxt(file_path, delimiter=',').tolist()]  # Load cache
            t = int(eval_scores[-1][0])  # Re-instate the largest t value recorded in the cached values
            last_eval = t  # This is also the last time we made a model eval call
            exp_schedule.update(t)  # Update the epsilon obj before passing into the method below
            # Before we continue training, populate the replay buffer and tracking variables with frames
            self._populate_buffer(replay_buffer, exp_schedule, episode_rewards, max_q_values, q_values,
                                  self.config["hyper_params"]["learning_start"])

            if self.config["model_training"].get("record"):  # If recording episodes, adjust episode counter
                # based on the largest cached value in the recordings directory so that any future recordings
                # continue to be numbered higher than the existing recording numbers
                file_names = [x for x in os.listdir(self.record_dir)]
                if len(file_names) == 0:  # If there are no existing recordings, set the episode count to 0
                    self.env.episode_id = 0
                else:  # Otherwise use the largest one we can find among them +1
                    self.env.episode_id = max([int(x.replace("game-", "").replace(".mp4", ""))
                                               for x in file_names]) + 1
        else:  # If no eval_scores.csv cached on disk, then create a new list to store values
            eval_scores = []
        # Compile a list of evaluation scores, begin with an eval score run with the model's current weights
        eval_scores.append((t, self.evaluate(num_episodes=self.config["model_training"]["num_episodes_test"],
                                             record_last=self.config["model_training"]["record"])))

        prog = Progbar(target=self.config["hyper_params"]["nsteps_train"])  # Training progress bar

        # 3). Interact with the environment, take actions, get obs + rewards, and update network params
        while t < self.config["hyper_params"]["nsteps_train"]:  # Loop until we reach the global training
            # step limit across all episodes, keep running episodes through the env until the limit is reached

            episode_reward = 0  # Track the total reward from all actions during the episode
            state = self.env.reset()  # Reset the env to begin a new training episode
            replay_buffer.add_entry(state)  # Add the starting board to the replay buffer

            while True:  # Run an episode of obs -> action -> obs -> action in the env until finished which
                # happens when either 1). the episode has been terminated by the env 2). the episode has
                # been truncated by the env or 3). the total training steps taken exceeds nsteps_train

                t += 1  # Increment the global training step counter i.e. every step of every episode +1

                # Update the exploration rate, learning rate and beta as we go, update them for the current t
                exp_schedule.update(t)
                lr_schedule.update(t)
                beta_schedule.update(t)

                # Compute the best_action (an int) to take from the current board position and also generate
                # the q_vals associated with each potential action that is available. Utilize the epsilon
                # exploration parameter, this does not require tracking gradients since we're interacting
                # with the env to generate data rather than computing a gradient update.
                if np.random.rand() < exp_schedule.param:  # With probability epsilon, choose a random action
                    action, state_value, action_values = self.env.action_space.sample(), 0, np.zeros(0)
                else:  # Otherwise actually use the model to evaluate
                    action, state_value, action_values = self.get_best_action(state)

                # Store the q values from the search process in the deque data structures
                if len(action_values) > 0:
                    max_q_values.append(np.max(action_values))  # Keep track of the max q-value returned
                    q_values.append(np.mean(action_values))  # Keep track of the avg q-value returned

                # Perform the selected action in the env, get the new state, reward, and stopping flags
                new_state, reward, terminated, truncated = self.env.step(action)
                reward = np.clip(reward, -1, 1)  # We expect +/-1, but add reward clipping for safety

                # Record the new state after taking this action in the replay buffer
                replay_buffer.add_entry(new_state)

                # Track the total reward throughout the full episode
                episode_reward += reward

                if t <= self.config["hyper_params"]["learning_start"]: # Populate the replay buffer
                    if t % self.config["model_training"]["log_freq"] == 0:  # Update logging every so often
                        learning_start = self.config['hyper_params']['learning_start']
                        sys.stdout.flush()
                        sys.stdout.write(f"\rPopulating the replay buffer {t}/{learning_start}...")
                        prog.reset_start()

                else: # Otherwise if the warm-up period has ended, then potentially perform gradient updates,
                    # performance logging and evaluations

                    # Perform a training step using the replay buffer to update the network parameters
                    loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.param,
                                                           beta_schedule.param)

                    if t % self.config["model_training"]["log_freq"] == 0:  # Update logging every so often
                        self.update_averages(episode_rewards, max_q_values, q_values, eval_scores)
                        self.add_summary(loss_eval, grad_eval, t)
                        max_R = np.max(episode_rewards) if episode_rewards else 0

                        prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg_R", self.avg_reward),
                                                  ("Max_R", max_R),
                                                  ("eps", exp_schedule.param),
                                                  ("Grads", grad_eval), ("Max_Q", self.max_q),
                                                  ("lr", lr_schedule.param)],
                                    base=self.config["hyper_params"]["learning_start"])

                    # If it has been more than eval_freq steps since the last time we ran an eval then run now
                    if (t - last_eval) >= self.config["model_training"]["eval_freq"]:
                        last_eval = t  # Record the training timestep of the last eval (now)
                        eval_scores.append(
                            (t, self.evaluate(num_episodes=self.config["model_training"]["num_episodes_test"],
                                              record_last=self.config["model_training"]["record"]))
                        )  # Compute a new eval score value for the model
                    save_eval_scores(eval_scores, self.config["output"]["plot_output"])  # Save results

                state = new_state  # Update for next iteration, move to the new FEN state representation
                # End the episode if one of the stopping conditions is met
                if terminated or truncated or t >= self.config["hyper_params"]["nsteps_train"]:
                    break

            # Perform updates at the end of each episode
            episode_rewards.append(episode_reward)  # Record the total reward received during the last episode
            ep_record = create_ep_record(self.env.board.move_stack)
            ep_record["t"] = t  # Add the training timestap as well to the logged info
            self.update_ep_history(ep_record, prefix="eval")  # Record game summary in the csv cache

        # Final screen updates
        self.logger.info("Training done.")
        self.save()  # Save the final model weights after training has finished
        eval_scores.append((t, self.evaluate(num_episodes=self.config["model_training"]["num_episodes_test"],
                                             record_last=self.config["model_training"]["record"])))
        save_eval_scores(eval_scores, self.config["output"]["plot_output"])

        duration = time.time() - start_time
        self.logger.info(f"Training finished in {duration / 60:.1f} minutes")

    def train_step(self, t: int, replay_buffer: ReplayBuffer, lr: float, beta: float) -> None:
        """
        Perform 1 training step to update the trainable network parameters of the self.v_network.

        :param t: The timestep of the current iteration.
        :param reply_buffer: A reply buffer used for sampling recent env transition observations.
        :param lr: A float denoting the learning rate to use for this update.
        :param beta: A hyperparameter used in prioritized experience replay sampling.
        :return: None.
        """
        loss_eval_total, grad_eval_total = 0, 0

        # Perform a training step parameter update
        learning_start = self.config["hyper_params"]["learning_start"]  # Warm-up period
        learning_freq = self.config["hyper_params"]["learning_freq"]  # Frequency of q_network param updates
        learning_updates = self.config["hyper_params"]["learning_updates"]  # How many update batches to run

        if t >= learning_start and t % learning_freq == 0:  # Update the q_network parameters with samples
            # from the reply buffer, which we only do every so often during game play
            for i in range(learning_updates):  # Run potentially many update steps
                loss_eval, grad_eval = self.update_step(replay_buffer, lr, beta)
                loss_eval_total += loss_eval
                grad_eval_total += grad_eval

        # Occasionally save the model weights during training
        if t % self.config["model_training"]["saving_freq"] == 0:
            self.save()

        return loss_eval_total / learning_updates, grad_eval_total / learning_updates

    def update_step(self, replay_buffer: ReplayBuffer, lr: float, beta: float) -> Tuple[int, int]:
        """
        Performs an update of the self.v_network parameters by sampling from replay_buffer.

        :param replay_buffer: A ReplayBuffer instance where .sample() gives us batches.
        :param lr: The learning rate to use when making gradient descent updates to self.v_network.
        :param beta: A hyperparameter used in prioritized experience replay sampling.
        :return: The loss = (y - y_hat)^2 and the total norm of the parameter gradients.
        """
        batch_size = self.config["hyper_params"]["batch_size"]

        # 1). Sample from the reply buffer to get recent states (encoded a FEN strings)
        state_batch, wts, indices = replay_buffer.sample(batch_size, beta)
        # [state_batch, wts, indices] -> (batch_size, ) lists of values

        # 2). Check that required components are present
        assert self.v_network is not None, "WARNING: Networks not initialized. Check initialize_model"
        assert self.optimizer is not None, "WARNING: Optimizer not initialized. Check add_optimizer"

        # 3). Zero the tracked gradients of the v_network model
        self.optimizer.zero_grad()

        # 4). Run a forward pass of the batch of states through the v_network and generate value estimates
        # i.e. what the v_network estimates is the discounted future return of following an optimal policy
        # from each state as the initial starting state
        v_est = self.v_network(state_batch).reshape(-1)  # Returns a torch.Tensor on self.device

        # 5). Compute the TD target values using a search function to get more accurate values by looking
        # ahead, we want to train the network to estimate this search function in the forward pass
        best_actions, v_search_estimates, action_values = self.search_func(state_batch)

        # 6). Compute gradients wrt to the MSE Loss function
        # Convert the inputs for this calculation to torch.Tensor and move to self.device
        wts = torch.from_numpy(wts).to(self.device)
        v_search_estimates = torch.from_numpy(v_search_estimates).to(self.device)

        loss, td_errors = self.calc_loss(v_est, v_search_estimates, wts)  # td_errors is a np.array
        replay_buffer.update_priorities(indices, td_errors)  # Update the priorities for the obs sampled
        loss.backward()  # Compute gradients wrt to the trainable parameters of self.v_network

        total_norm = torch.nn.utils.clip_grad_norm_(self.v_network.parameters(),
                                                    self.config["model_training"]["clip_val"])

        # 7). Update parameters with the optimizer by taking a step
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()  # Perform a gradient descent update step for all parameters

        return loss.item(), total_norm.item()  # Return the loss and the norm of the gradients

    def evaluate(self, num_episodes: int = None, record_last: bool = True, verbose: bool = True) -> float:
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
        self.logger.info(f"\nRunning Evaluation for {num_episodes} episodes")
        max_moves = self.config["model_training"]["max_eval_moves"]
        all_losses = []

        with logging_redirect_tqdm():
            for g in tqdm(range(num_episodes), ncols=75):
                # Compute the losses for each move for 1 self-play game
                ep_losses, move_stack = evaluate_agent_game(self.agent_move_func, max_moves=max_moves,
                                                            return_move_stack=True)
                all_losses.extend(ep_losses)  # Aggregate the losses over all moves and all games
                self.update_ep_history(create_ep_record(move_stack), prefix="eval")  # Record game summary
                if verbose is True:
                    msg = f"Game {g + 1}: ACPL = {sum(ep_losses) / len(ep_losses):.1f}, {len(move_stack)}"
                    msg += " moves total"
                    self.logger.info(msg)

        overall_acpl = sum(all_losses) / len(all_losses)
        self.logger.info(f"\nOverall ACPL: {overall_acpl:.1f}")

        if record_last is True:  # Save the last evaluation episode as a recording
            output_dir = os.path.join(self.record_dir, f"game-{self.env.episode_id}")
            os.makedirs(output_dir, exist_ok=True)  # Create an output folder for this game recording
            save_recording(move_stack, os.path.join(output_dir, f"game-{self.env.episode_id}.mp4"))
            save_move_stack(move_stack, os.path.join(output_dir, f"game-{self.env.episode_id}.txt"))

        return overall_acpl

    def update_ep_history(self, ep_record: Dict, prefix: str = "train") -> None:
        """
        Appends the info of the input ep_record to an existing {prefix}_ep_history.csv if one exists,
        otherwise this method creates a new csv file and saves down the episode record summary as the first
        result. This method is generally called after running a fulling training or evaluation episode and is
        used to add episode summary entries to the output cache.

        :param ep_record: An episode record recorded as a dictionary.
        :param prefix: A prefix value for the {prefix}_ep_history.csv file.
        :return: None, updates the local {prefix}_ep_history.csv cache or creates one.
        """
        ep_record = pd.Series(ep_record)
        filename = "ep_history.csv" if not prefix else f"{prefix}_ep_history.csv"
        ep_df_file_path = os.path.join(self.config["output"]["plot_output"], filename)
        if os.path.exists(ep_df_file_path):  # Check if an {prefix}_ep_history file has been cached
            ep_df = pd.read_csv(ep_df_file_path)  # Read in the data from disk
            ep_record["episode_id"] = len(ep_df)
            ep_df.loc[len(ep_df), :] = ep_record
        else:  # If not, then use this ep_record as the first entry in the ep_df
            ep_df = ep_record.to_frame().T

        for col in ["outcome", "winner", "end_state"]:  # Change these columns to string format
            ep_df[col] = ep_df[col].astype(str)

        ep_df.to_csv(ep_df_file_path, index=False)  # Write out a new copy of the ep_df to disk to cache

    def record(self, max_moves: int = 300) -> None:
        """
        This method records a video for 1 episode using the model's current weights and saves it to disk.
        Videos are saved to self.config["output"]["record_path"] and are named: game-{ep} where ep is the
        episode counter from the environment which will monotonically increase during training.

        :param max_moves: An integer denoting the max number of moves allowed per game.
        :return: None. Records a video of a self-play game and saves it to disk.
        """
        state = self.env.reset()  # Clear out any existing game state and begin fresh
        self.env.record = True  # Toggle on the recording setting within the ChessEnv
        # Create a policy function that return the best move index according to the model
        policy = self.policy()

        move_counter = 0
        terminated, truncated = False, False
        while move_counter < max_moves and not (terminated or truncated):
            new_state, reward, terminated, truncated = self.env.step(policy(state))
            state = new_state  # Update for next iteration
            move_counter += 1

        state = self.env.reset()  # The env reset triggers the steps in the move_stack to be recorded
        self.env.record = False  # Turn off recordings for future env steps
