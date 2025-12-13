"""
This module creates a simplified test environment used for testing and debugging.
"""

import numpy as np
from typing import Tuple, Optional


class ActionSpace:
    def __init__(self, n: int):
        """
        Action space with n possible actions an agent can select.
        """
        self.n = n  # Record the size of the action space

    def sample(self):
        """
        Randomly sample an action from the full action space.
        """
        return np.random.randint(0, self.n)


class ObservationSpace:
    def __init__(self, n: int, shape: Tuple[int]):
        """
        An observation space with states represented in self.states where each is of size shape and filled
        with random integers from a specific interval range (non-overlapping with other states).

        :param n: The number of distinct states to create i.e. len(self.states).
        :param shape: The shape of each state stored in self.states.
        """
        self.shape = shape  # Record the size of each state representation
        # Randomly generate n states as integer valued np.ndarrays with different ranges of ints to make them
        # easily differentiable by a model
        lower, upper = 0, 10  # The lower and upper bound for each state's random ints
        self.states = []
        for i in range(n):
            self.states.append(np.random.randint(lower, upper, shape, dtype=np.uint16))
            lower, upper = upper, upper + 10  # Increment as we go to make each state easily identifiable


class EnvTest:
    """
    A lightweight test environment for debugging and small scale testing.

    The action space and observation space are of equal size. The observation space (states) are fully
    connected and state transitions are deterministic i.e. from any starting state, the agent can reach any
    other state of the agent's choice by taking action A which will move it to the state indexed by A e.g.
    action=3 will move the agent into the state indexed at 3.

    There is a fixed reward that is received by the agent when it moves into a new state recorded in the
    self.rewards property. When the agent takes action X and moves from state S to X, it receives reward
    self.reward[X] unless the current state S is the max reward state, in which case the reward will be
    (-5) * self.reward[X].
    """

    def __init__(self, n: int, shape: Tuple[int] = (84, 84, 3), seed: Optional[int] = None):
        """
        Instantiates the EnvTest object.

        :param n: The number of states and actions.
        :param shape: The shape of each input state.
        """
        if seed is not None:  # If a seed has been specified, then use it to init the test env
            np.random.seed(seed)
        self.n = n  # Record the size of the state and action space
        self.rewards = np.random.normal(0, 1, n)  # Generate rewards for arriving in each state
        print("TestEnv.rewards", self.rewards)
        self.cur_state = 0  # Index of the current state in self.observation_space.states
        self.num_iters = 0  # Number of steps taken so far in this episode
        self.max_reward_state = np.argmax(self.rewards)  # Record which state gives the highest reward
        self.action_space = ActionSpace(n)  # Record the action space object
        self.observation_space = ObservationSpace(n, shape)  # Record the obs space object

    def reset(self):
        self.cur_state = 0  # Reset back to the first state indexed at 0
        self.num_iters = 0  # Reset the n_iter counter within the current episode back to 0
        self.was_in_second = False
        return self.observation_space.states[self.cur_state]  # Return the starting state

    def step(self, action: int) -> tuple:
        """
        Takes an input action from the agent and runs 1 timestep in the environment and returns a tuple of:
            (new_state, reward, is_terminated, is_truncated, info)

        :param action: An input action from the agent to take from the current state.
        :return: A tuple of (new_state, reward, is_terminated, is_truncated, info).
        """
        assert 0 <= action < self.n, "Action value outside of acceptable range"
        self.num_iters += 1  # Increment the number of iterations taken within the current episode
        # Special Rule: When we take an action from the max reward state, the next state's reward is -5x
        reward = 1.0 if self.cur_state != self.max_reward_state else -5.0
        self.cur_state = action  # Taking action A moves the agent to the state indexed by A
        reward *= self.rewards[self.cur_state]  # Record the reward received by moving to the new state
        # Return a tuple of (new_state, reward, is_terminated, is_truncated, info) to match a gym env
        output = (self.observation_space.states[self.cur_state], reward, self.num_iters >= 10,
                  False, {"ale.lives": 0},)
        return output

    def render(self) -> None:
        """
        Reports the current state.
        """
        print(self.cur_state)
