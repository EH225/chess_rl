"""
This module defines the replay buffer which is used to store experiences of the agent as it interacts with
the environment and allows for a sampling of past experiences for training steps.
"""

import numpy as np
import torch
from typing import Optional, List


class ReplayBuffer:
    """
    A memory-efficient implementation of a replay buffer that records chess game states using a FEN
    (Forsyth–Edwards Notation) representation of the game state denoting the location of the pieces on the
    board, who is to move next, who can castle, en passant etc.
        E.g. 'r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4'

    A replay buffer acts as a recent memory bank for recent observations from the env gathered by the RL
    agent. By caching these values, we can randomly sample from them to generate low-correlation observation
    samples during gradient-based training updates to the model parameters that are not too heavily anchored
    toward recent env experiences.

    A replay buffer (also called experience replay) is a core component of training that significantly
    improves learning stability and efficiency by reducing the correlation between consecutive samples,
    improves data efficiency by reusing obs from the env multiple times, smooths out the learning process,
    enables off-policy learning, and helps to avoid feedback loops during training.

    For this implementation, search functions are used online to select actions and compute TD target values.
    Therefore instead of storing (s, a, r, a', truncated, terminated) as one usually would, it suffices to
    store only s the starting game state since the dynamics model of the env (chess) is know and accessible.

    Game states are recorded interally as a list of strings.

    This implementation also allows for Prioritized Experience Replay (PER). PER assigns a priority to each
    transition—typically based on the TD error (the abs difference between predicted and target), and
    therefore samples the high-error transitions more often to help the RL agent learn faster by focusing on
    experiences where it struggled the most. This can help improve the speed of training and also the quality
    of the final trained model. Certain rate, but infrequent observations can be overshadowed by abundant,
    but uninformative transitions.
    """

    def __init__(self, size: int, eps: float = 1e-5, alpha: float = 0.6, seed: Optional[int] = None):
        """
        The max state capacity of the replay buffer is specified by the size input. Game states are
        represented by their FEN encoding.

        eps and alpha are hyperparameters related to the Prioritized Experience Replay (PER) sampling.

        :param size: The max number of states to record in the buffer. When the buffer size is full and a new
            entry is added, the oldest obs are overwritten, FIFO inventory management is used.
        :param eps: The epsilon value to use when updating priority values.
        :param alpha: Used to control the degree of priority sampling. When alpha=0, then no priority
            sampling is performed, all indices have an equal change of being sampled. When alpha=1, then we
            have maximal prioritization of the larger TD error obs.
        :param seed: An optional seed that can be set which controls the selection of random samples.
        """
        self.size = int(size)  # Record the max capacity of the replay buffer

        self.last_idx = None  # Track the last index at which a new frame (s) was written
        self.next_idx = 0  # Tracks the next index to write historical observational data to
        self.num_in_buffer = 0  # Tracks how many observations are currently saved in the buffer, <= size
        self.buffer_full = False  # Set to True when the buffer reaches full capacity

        # Init variables to store the info for each transition observation
        self.states = [0 for i in range(size)]  # Current states (s) as FEN encodings
        self.priority = np.zeros(size)  # The |TD error| + eps, for prioritized experience replay
        self.eps = float(eps)  # Epsilon value for computing priority scores i.e. p = (td_err + eps) ** alpha
        self.alpha = float(alpha)  # Controls how much more we sample high priority obs, alpha=0 equal prob
        self.max_priority = float(eps)  # The priority values are all initialized at eps

        self.seed = seed  # Store the random seed provided if any
        self.rng = np.random.default_rng(seed)  # Create a random number generator for sampling with a seed

    def _get_next_idx(self, idx: int) -> int:
        """
        Given an input idx, this function returns the next index with wrap around i.e. at idx == size - 1,
        the next value index is 0. This is the inverse function of _get_prior_idx.

        :param idx: An input index in the range [0, self.size-1].
        :return: The next sequential index which is either idx + 1 or 0.
        """
        return (idx + 1) % self.size

    def _get_prior_idx(self, idx: int) -> int:
        """
        Given an input idx, this function returns the prior index with wrap around i.e. 1 -> 0 and at the
        start we get 0 -> size - 1. This is the inverse function of _get_next_idx.

        :param idx: An input index in the range [0, self.size - 1].
        :return The prior sequential index which is either idx - 1 or self.size - 1.
        """
        return (idx - 1) % self.size

    def add_entry(self, state: str) -> int:
        """
        Stores a single game state in FEN encoding in the replay buffer at the next available index and
        overwriting old frames if necessary using a FIFO management system.

        :param state: A FEN string value encoding the current state of the chess game.
        :return: An integer index designating the location where the frame was stored internally.
        """
        self.states[self.next_idx] = state  # Record in the replay buffer at the next write location
        self.last_idx = self.next_idx  # Record the index where this new frame was written to
        self.next_idx = self._get_next_idx(self.next_idx)  # Update next_idx to the next write location, wrap
        # around back to 0 at the beginning again if we reach the end
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)  # Update the number of elements in the
        # buffer which is capped at self.size, but is generally 1 larger than before
        self.buffer_full = (self.num_in_buffer == self.size)  # Track if the buffer has been fully filled
        return self.last_idx  # Return the index in the replay buffer where this frame was stored

    def add_entries(self, state_batch: List[str]) -> None:
        """
        Adds a list of input states to the replay buffer i.e. this method is equlivalent to calling add_entry
        for each state in the input state_batch.

        :param state_batch: An input list of FEN string value encoding the current state of the chess game.
        :return: None.
        """
        assert isinstance(state_batch, list), "state_batch must be a list"
        for state in state_batch:
            self.add_entry(state)

    def sample(self, batch_size: int, beta: float = 0.1) -> List[torch.Tensor]:
        """
        This method randomly samples batch_size starting state observations from the replay bugger.
        If there are no sufficiently many observations in the reply buffer, then an error is raised.

        :param batch_size: The number of randomly sampled historical examples to return from the buffer.
        :param beta: Because PER distorts the real data distribution, we use importance sampling to un-bias
            our gradient updates which is controlled by this beta parameter which should be annealed from
            small to large during training.
        :returns:
            state_batch: A list of state FEN strings
            wts: A np.ndarray of weight values associated with the sampled observations.
            indices: The indices of the sampled states.
        """
        assert isinstance(batch_size, int) and batch_size >= 1, "batch_size must be an int >= 1"
        assert batch_size <= self.num_in_buffer, "batch_size exceeds number of examples in the buffer needed"
        # Randomly sample indices of s to include in this batch in the range [0, num_in_batch - 1]
        if self.alpha > 0:  # Use priority sampling, when alpha == 0, then we're using uniform sampling
            probs = (self.priority[:self.num_in_buffer] + self.eps) ** self.alpha  # Compute the priority wts
            probs /= probs.sum()  # Normalized to be a probability vector
            # indices = self.rng.multinomial(batch_size, probs)  # Samples with replacement but that's okay
            indices = self.rng.choice(len(probs), size=batch_size, replace=False, p=probs)
            wts = (1 / (probs[indices] * self.num_in_buffer)) ** beta  # Extract the relevant weights
            wts /= wts.max()  # Normalize by the max of weights to prevent extreme gradients
        else:  # Otherwise, use naive sampling where all indices have an equal change of being selected
            indices = self.rng.choice(np.arange(0, self.num_in_buffer), size=batch_size, replace=False)
            wts = np.ones(batch_size)  # Set weights equal for all samples with weights of 1

        state_batch = [self.states[i] for i in indices]
        return state_batch, wts, indices

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        """
        Updates the priority scores associated with the indices provided, which determined how likely a given
        observation is to be sampled during training i.e. the larger the |td_error|, the more likely an obs is
        to be selected during sampling.

        :param indices: The indices associated with the |td_errors| provided.
        :param td_errors: A tensor of absolute TD errors i.e. |Q(s, a) - (r + gamma * max(Q(s')))|
        :returns: None, modifies the internal data structure holding the priorities for each obs.
        """
        priorities = (td_errors + self.eps) ** self.alpha  # Compute updated priority values from TD diffs
        self.priority[indices] = priorities  # Update values internally
        self.max_priority = max(self.max_priority, priorities.max())  # Update the max globally priority
