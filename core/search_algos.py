"""
This module builds off the classes contained in core/base_components and defines a class instance for a linear
deep Q-network and a CNN-based deep Q-network that replicates the network archtecture of the Google DeepMind
model published in Nature.
"""
from __future__ import annotations
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import numpy as np
import torch, chess
import torch.nn as nn
from utils.chess_env import ChessEnv
from typing import Tuple, List, Callable, Optional

#########################
### Naive Search Algo ###
#########################
# TODO: Section marker

def naive_search(state: str, model, gamma: float = 1.0, batch_size: int = 64, **kwargs
                 ) -> Tuple[int, float, np.ndarray]:
    """
    Performs a naive search operation where the next action selected it the one which results in the highest
    estimated next state value according to the model. This function performs a depth 1 forward search.

    :param state: A FEN string denoting the current game state.
    :param gamma: The temporal discount factor to apply to distant rewards. For chess this is usually 1.0.
    :param model: A value function approximator that produces board value estimates from the perspective of
        the players whose turn it is with model(state_batch) where state_batch is a list of FEN strings.
    :param batch_size: The size of the state_batch (a list of strings) fed to model.
    :return:
        - best_action (int): The best action found in the search process.
        - state_value (float): The estimated value of the input starting state.
        - action_values (np.ndarray): The estimated value of each possiable next action.
    """
    env = ChessEnv(initial_state=state)  # Instantiate the current game state
    if env.ep_ended:  # If the episode has already ended, then there is no searching over action to be done
        # The value is always computed from the perspective of the player who is to move next in the state,
        # if the game has ended in a checkmate, then the last move from the opponent achieved it and the
        # player to move next has lost, return a value of -1 and None for best_action and action_values
        value = -1 if env.board.is_checkmate() else 0
        return None, value, None

    # Else: Perform a search over possiable next actions in the env from the current starting state
    action_values = [] # Record a value estimate for each possiable next action from the starting state
    state_batches = [] # Record the FEN string encodings of possiable next states that we will feed through
    # the model in batches to get out value estimates for each
    state_batches_idx = []  # Record the indices of the actions each state in state_batches corresponds to,
    # we will only run states that are non-terminal through the model to minimize computations i.e. if the
    # state is terminal, no value estimate is required, the outcome is given directly from the env

    for i in range(env.action_space.n):  # Consider all possiable legal moves
        new_state, reward, terminated, truncated = env.step(i)  # Make the candidate move
        action_values.append(reward)  # Record the immediate reward of this action
        if not terminated:  # Check if the game reached a terminal state, if so then bootstrap the value,
            # this includes a truncated state from which we could continue to play additional moves
            # estimation thereafter using the value approximator (model)
            if len(state_batches) == 0 or len(state_batches[-1]) == batch_size:
                state_batches.append([new_state, ]) # Start a new list for this next state obs
            else: # Otherwise, append to the existing batch so that each is batch_size or less
                state_batches[-1].append(new_state)
            state_batches_idx.append(i)  # Make note of the index of action_values to add to later
        env.rev_step() # Backtrack so that we can try a different move instead

    # Compute the bootstrap value estimates in batches for the next possiable states after each action
    value_estimates = []
    for state_batch in state_batches: # Compute the value estimates in batches to minimize runtime
        with torch.no_grad(): # Disable grad-tracking, not needed since no gradient step being taken
            v_est = model(state_batch).cpu().tolist()  # Move the data from torch to a list on the CPU
        value_estimates.extend(v_est)  # Aggregate the state value estimates into 1 linear list

    for idx, val in zip(state_batches_idx, value_estimates):  # Add the bootstrapped value estimates for s'
        # gamma is applied to discount future rewards and the value estimates need to have their signs
        # reversed since they are computed from the perspective of the other player i.e. the one who plays
        # from s', chess is a zero-sum game so the value to one is the negative of the value to the other
        action_values[idx] += gamma * val * (-1)

    # Determine the best action and the highest value by computing the argmax and max
    action_values = np.array(action_values)
    best_action, state_value = action_values.argmax(), action_values.max()
    return best_action, state_value, action_values

######################################
### Minimax Alpha-Beta Search Algo ###
######################################
# TODO: Section marker

def _minimax_search(state: str, model, cache: dict, alpha: float = -float("inf"), beta: float = float("inf"),
                    gamma: float = 1.0, depth: int = 0, max_depth: int = 3, maximize: bool = True
                    ) -> Tuple[int, float, np.ndarray]:
    """
    Recursive helper function for implementing minimax search with alpha-beta pruning.

    Minimax search performs an exhaustive search through the game tree to evaluate all possible moves from
    the current starting position until either a terminal state is reached or a max depth, at which point a
    value approximator (model) is used instead. All rewards are modeled from the perspective of the initial
    player i.e. the maximizer. The maximizer and minimizer take turns up the tree selecting a max or min value
    among all options at each node.

    Alpha-Beta pruning is a technique to help reduce the amount of computation in minimax search by pruning
    branches of the search tree that are unnecessary given what has already been explored. At a given node
    evaluation, alpha is a lower bound for what the maximizer can get from selecting another node at the same
    level and beta is the upper bound for what the minimizer can get from selecting another node at the same
    level. E.g. Say the maximizer is evaluating a node with 3 child nodes. The first child node yields a value
    from the minimzer of 3. Then when evaluating the second child node, the maximizer will be looking for
    something that is >= 3. If during the expansion of that node, we find that a value of 2 is an option for
    the minimizer at that node, that will be an upper bound for anything the minimizer chooses for that node.
    Therefore, we can stop expanding branches along the second child node since we now know that max value of
    the node will be 2 and the maximizer will always prefer the first child node over it instead.

    If the input state is a terminal state or max_depth == 0, then (state_val, nodes_evaluated) will be
    returned, otherwise (action_vals, nodes_evaluated) will be returned where action_vals is a list of
    estimated state values for each possible action from the current initial state which can be used to find
    the best action and overall state value estimate.

    :param state: A FEN string denoting the current game state.
    :param model: A value function approximator that produces board value estimates from the perspective of
        the players whose turn it is with model(state_batch) where state_batch is a list of FEN strings.
    :param cache: A dictionary-like data structure used for caching values from model to reduce repeat calcs.
    :param alpha: A lower bound for what the maximizer can get from selecting another node at the same level.
    :param beta: An upper bound for what the minimizer can get from selecting another node at the same level.
    :param gamma: The temporal discount factor to apply to distant rewards. For chess this is usually 1.0.
    :param depth: The depth of the node currently being evaluated. The root node has a depth of 0. Intuitively
        depth is a measure of how many moves (black and white) have been made since the initial state.
    :param max_depth: This limits the max depth of nodes in the search tree. When depth==max_depth, then
        model(state) is used to estimate the value of the state instead of further searching.
    :param maximize: A bool indicating whether the player looking at the current node is trying to maximize.
    :return:
        - action_vals (List[int] or int): A list of state value estimates for each action or 1 float value.
        - nodes_evaluated (int): The total number of nodes evaluated in the search tree.
    """
    assert isinstance(max_depth, int) and max_depth >= 0, "max_depth must be >= 0 and an integer"
    env = ChessEnv(initial_state=state)  # Instantiate the current game state

    if env.ep_ended: # Check if this is a terminal state, if checkmate then the current player lost
        return ((-1 if maximize else 1) if env.board.is_checkmate() else 0), 1

    elif depth == max_depth:  # If this node is at the maximal depth, then use the model to estimate the value
        state = " ".join(state.split()[:-1])  # Remove the full move counter from the state FEN string
        if state in cache:  # Check if this state has already been predicted using the model
            val = cache[state] # If so, then use the pre-computed cached value to save compute
        else:  # Otherwise, if not yet computed, then compute and cache the output of the model
            val = model([state, ]).cpu().tolist()[0]
            cache[state] = val
        return (val * (1 if maximize else -1)), 1

    else:  # Otherwise continue to explore moves in the search tree recursively
        nodes_evaluated = 0  # Track the total number of nodes evaluated (leaf + middle)
        v = alpha if maximize is True else beta
        if depth == 0:  # At the root node, return a list of action values instead of a float
            action_vals = []

        for action in range(env.action_space.n):
            next_state, reward, terminated, truncated = env.step(action)
            reward = reward * (1 if maximize else -1)  # Flip if needed for the perspective of the maximizer

            if maximize is True: # A turn from the perspective of the original player who will want to
                # maximize the overall value of the game
                if not terminated:  # If the game continues further, recursively evaluate
                    val, n = _minimax_search(next_state, model, cache, v, beta, gamma, depth + 1,
                                             max_depth, False)
                    v = max(v, reward + gamma * val)  # Update the value of this node from the view of max
                else:  # Otherwise the value of this action is equal to the reward obtained
                    val, n = 0, 1  # No further states to explore, next_state is terminal
                    v = max(v, reward)

                if v >= beta:  # Use early stopping to prune branches. v is the estimate so far of what we
                    # can get as the maximizer from pathways beyond this current node. As the maximizer, we
                    # will select the highest value among the children. Therefore the current value of v is
                    # a lower bound as to what the maximizer will select. beta tells us that the minimizer
                    # can select at value of at least beta somewhere else at the same level, therefore it will
                    # never pick this node since the value will be at least v or higher as we maximize, thus
                    # we stop exploring the child nodes since this node will never be selected be min
                    return v, nodes_evaluated

            else:  # A turn from the perspective of the opposing player who will want to minimize the overall
                # value of the game
                if not terminated:  # If the game continues further, recursively evaluate
                    val, n = _minimax_search(next_state, model, cache, alpha, v, gamma, depth + 1,
                                             max_depth, True)
                    v = min(v, reward + gamma * val)
                else:  # Otherwise the value of this action is equal to the reward obtained
                    val, n = 0, 1  # No further states to explore, next_state is terminal
                    v = min(v, reward)

                if v <= alpha:  # Use early stopping to prune branches. v is the estimate so far of what we
                    # can get as the minimizer from pathways beyond this current node. As the minimizer, we
                    # will select the lowest value among the children. Therefore the current value of v is
                    # an upper bound as to what the minimizer will select. alpha tells us that the maximizer
                    # can select a value of at least alpha somewhere else at the same level, therefore it will
                    # never pick this node since the value will be at most v or lower as we minimize, thus
                    # we stop exploring the child nodes since this node will never be selected be max
                    return v, nodes_evaluated

            nodes_evaluated += n  # Track how many total nodes were evaluated in the recursive call
            if depth == 0:  # Record the estimated value of each action that can be taken
                action_vals.append(reward + gamma * val)
            env.rev_step() # Back track in the env for the next action
        nodes_evaluated += 1  # Add 1 more now that we've finished evaluating this root node

        return (v, nodes_evaluated) if depth > 0 else (action_vals, nodes_evaluated)


def minimax_search(state: str, model, gamma: float = 1.0, batch_size: int = 64, horizon: int = 3, **kwargs
                   ) -> Tuple[int, float, np.ndarray]:
    """
    Performs minimax-search with alpha-beta pruning up to a particular search horizon after which the model
    is used as a state value approximator to perform bootstrapping.

    See _minimax_search for details.

    :param state: A FEN string denoting the current game state.
    :param gamma: The temporal discount factor to apply to distant rewards. For chess this is usually 1.0.
    :param model: A value function approximator that produces board value estimates from the perspective of
        the players whose turn it is with model(state_batch) where state_batch is a list of FEN strings.
    :param batch_size: The size of the state_batch (a list of strings) fed to model.
    :param hoirzon: The max depth of the search tree at which point the model is used to estimate values
        instead.
    :return:
        - best_action (int): The best action found in the search process.
        - state_value (float): The estimated value of the input starting state.
        - action_values (np.ndarray): The estimated value of each possiable next action.
    """
    # Reply on the helper function to run the minimax search with alpha-beta pruning
    action_vals, nodes_evaluated = _minimax_search(state=state, model=model, cache={}, alpha=-float("inf"),
                                                   beta=float("inf"), gamma=gamma, depth=0,
                                                   max_depth=horizon, maximize=True)
    if isinstance(action_vals, list):  # A list will be returned when state is non-terminal
        action_vals = np.array(action_vals)
        best_action, state_value = action_vals.argmax(), action_vals.max()
    else: # Otherwise, if the input state was terminal, then only a single float value will be returned by
        # the helper function denoting the state's reward for the player whose move it is next
        return None, action_vals, None

    return best_action, state_value, action_vals


### Create a little bit better of a board estimator, count the material differences
def material_heuristic1(state_batch: List[str]) -> torch.Tensor:
    output = torch.zeros(len(state_batch))
    val_dict = {"p":1, "n":3, "b":3, "r":5, "q": 10, "k":0}
    for i, s in enumerate(state_batch):
        board = chess.Board(s)
        net_material = 0
        for p in board.piece_map().values():
            piece_val = val_dict[p.symbol().lower()]
            if board.turn and p.symbol().islower():
                piece_val *= -1
            elif not board.turn and p.symbol().isupper():
                piece_val *= -1
            net_material += piece_val

        output[i] = net_material
    return output

####################################
### Monte Carlo Tree Search Algo ###
####################################
# TODO: Section marker

def material_heuristic(state: str) -> float:
    """
    Computes a heuristic evaluation of board value based on net material from the perspective of the player
    whose turn is next according to the FEN state encoding.

    This is computed as the net material difference normalized to be [-1, 1] where each piece is worth:
        pawn (1), knight (3), bishop (3), rook (5), queen (9), king (0)

    The net material sum is divided by 39 since 1*8 + 3*2 + 3*2 + 5*2 + 9*1 = 39 which is close to the max
    possible material difference i.e. one side has a only a king, the other has all starting pieces. Values
    are also clipped incase a players obtains additional high-value pieces e.g. queens via pawn promotion.

    :param state: A FEN string denoting the current game state.
    :return: A float value representing the approximate value of the board for the player to move next.
    """
    piece_values = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}
    board = chess.Board(state)  # Convert to a chess.Board object to access the piece map
    net_material = 0
    for p in board.piece_map().values():
        piece_val = piece_values[p.symbol().lower()]  # Get the absolute value of each piece
        if board.turn is True and p.symbol().islower(): # For white, lower-case pieces are foes
            piece_val *= (-1)
        elif board.turn is False and p.symbol().isupper(): # For black, upper-case pieces are foes
            piece_val *= (-1)
        net_material += piece_val

    # 39 is the largest net material difference we expect, but clip at [-1, +1] to avoid outliers
    return np.clip(net_material / 39, -1, 1)


class Node:
    """
    Class object used in Monte Carlo Tree Search (MCTS).
    """
    def __init__(self, state: str, parent: Optional[Node]):
        """
        Instantiates a MCTS tree node.

        :param state: A FEN string denoting the current game state.
        :param parent: A pointer to this node's parent node in the search tree.
        """
        self.state = state  # Record the FEN state encoding of this game state
        self.state_ = " ".join(state.split()[:-1])  # Remove the total move counter from the FEN
        board = chess.Board(state)  # Init a chess board obj for internal ops, skip recording to save memory
        self.parent = parent  # Record a pointer to this node's parent node, will be None if this is the root
        # Make note of whose turn it is (white or black) at the root node. If parent is None, then this is a
        # root node, record from the current game state, otherwise inherit from this from the parent node
        self.p_turn = board.turn if parent is None else parent.p_turn
        self.node_turn = board.turn  # Record the turn value for this game state's next-to-move
        self.value_sum = 0  # Record the sum of all value updates to this node, starts at 0
        self.n_visits = 1 if parent is None else 0  # Record the total node visits, root begins with 1
        self.is_expanded = False  # Bool flag for if this node has been expanded yet, beings at False

        # Bool flag indicating whether this is a terminal game state node
        self.is_terminal = board.is_game_over()
        if self.is_terminal: # If the game is over, record the terminal reward value from the view of p_turn
            # If the game ends in checkmate and p_turn is next to move, then p_turn lost
            self.terminal_reward = (-1 if board.turn is self.p_turn else 1) if board.is_checkmate() else 0
        else:
            self.terminal_reward = None

        self.virtual_loss = 0  # Used to discourage the same node being selected many time during batched
        # leaf node selection, beings at 0, ticks up as more nodes in the batch select it
        self.children = []  # A collection of pointers to other child nodes, doesn't get populated until
        # this node is expanded i.e. visited for the first time by the selection step
        self.unvisited_leaf_nodes = 1  # Track how many unvisited leaf nodes including this node itself
        self.prior = 0  # Init the prior value as 0 unless otherwise updated by a heuristic
        self.c_init, self.c_base = 1.25, 19652  # Constants used for the exploration term c_puct

    def Q(self) -> float:
        """
        Returns the average of all rewards backpropagated though this node i.e. value_sum / n_visits.
        """
        return self.value_sum / self.n_visits if self.n_visits > 0 else self.prior

    def argmax_PUCT_child(self) -> Optional[Node]:
        """
        Uses Prior-Guided Upper Bound Confidence Intervals for Trees (PUCT) to select a child action with a
        virtual loss to discourage making the same selections multiple times.

        This method returns None if this node has no children (i.e. is a terminal state), otherwise it returns
        a pointer to the argmax child node according to the PUCT with virtual loss formula:

            PUCT = Q_eff(child) + c * P(child) * sqrt(N(parent)) / (1 + N_eff(child))

        where:
            - N_eff(child) = child.n_visits + child.virtual_loss i.e. an adj visit count for past selections
            - virtual_loss = A term that is used to reduce the same nodes being selected during batched
              selection so that state inputs can be fed into the value function approximator in batches
            - Q_eff(child) = child.sum_value / N_eff(child)
            - c = An exploration modulation constant, determines how much weight to put towards exploration
              with c = c_init + log((N_eff(child) + c_base + 1) / c_base)
            - P(child) = The prior value estimate for the child node, used to perform informed exploration
            - N(parent) = How many times the parent node has been visited in total
        """
        if not self.children or self.unvisited_leaf_nodes == 0: # Return None if there are no child nodes
            return None  # yet unexplored down this route

        # Look for the max PUCT value among all child nodes with unvisited leaf nodes
        argmax_node, max_val = None, -float("inf")
        for child in self.children:
            if child.unvisited_leaf_nodes == 0:  # Skip if everything has already been explored
                continue
            else:
                n_eff = child.n_visits + child.virtual_loss  # virtual_loss discouranges repeat selections
                val = child.value_sum / n_eff if n_eff > 0 else 0  # Avoid division by 0 issues
                c_puct = self.c_init + np.log(n_eff + self.c_base + 1) - np.log(self.c_base)
                val += c_puct * child.prior * np.sqrt(self.n_visits) / (1 + n_eff)
                if val > max_val:  # Update the argmax node seen so far
                    argmax_node, max_val = child, val

        return argmax_node

    def expand_legal_moves(self, prior_heuristic: Callable = None) -> None:
        """
        This method expands the current node i.e. adds unexplored child nodes to self.children if applicable
        (i.e. if this node is non-terminal) and sets self.is_expanded to True. The child nodes created have
        their prior values set by prior_heuristic(child.state) if provided, otherwise a uniform prior of
        1 / num_actions is used where num_actions == n_children.

        :param prior_heuristic: A heuristic function for computing a prior value for each child node added.
            Should be a function that takes the FEN state (str) as its first argument and returns the value
            estimate for the player whose move is next.
        :return: None, Node objects are added as children to the current node.
        """
        assert not self.is_expanded, "This node as already been expanded"
        unvisited_leaf_nodes_chg = 0  # Record how many new unexplored leaf nodes are created
        if not self.is_terminal: # If not terminal, then there are additional child nodes we can add
            board = chess.Board(self.state)  # Init a chess board object for internal operations
            uniform_prior = 1 / board.legal_moves.count()
            for move in board.legal_moves:  # Add a child node for each legal move starting here
                board.push(move) # Make this move on the board to get the next resulting game state
                child = Node(state=board.fen(), parent=self)
                if prior_heuristic is None:  # If no prior_heuristic provided, then set to the unif prior 1/n
                    child.prior = uniform_prior
                else:  # If a prior_heuristic is given, then use it to generate a prior for each child node
                    # Flip sign if eval from other side so that all are from the perspective of the root node
                    sign_flip = (1 if child.node_turn is child.p_turn else -1)
                    child.prior = prior_heuristic(child.state) * sign_flip
                self.children.append(child)
                board.pop() # Backtrack to visit the next legal move
                unvisited_leaf_nodes_chg += 1  # Record another unexplored child node added

            node = self  # Back propagate the update for unvisited_leaf_nodes to the root node
            while node is not None:
                node.unvisited_leaf_nodes += unvisited_leaf_nodes_chg
                node = node.parent

        self.is_expanded = True # Set this flag to true now that this node has been expanded with children

    def incriment_virtual_loss(self) -> None:
        """
        Starting at this node, this method incriments the virtual_loss counters for all nodes along the path
        from this node to the root node and also decriments the unvisited_leaf_nodes counters to prevent
        duplicate node selections from being made and to discourage similar node path selections overall.
        """
        node = self
        while node is not None:
            node.virtual_loss += 1
            node.unvisited_leaf_nodes -= 1
            node = node.parent  # Move up to the next node along the path to the root

    def backup(self, val_est: float) -> None:
        """
        Starting at this node, this method backs up the value estimate (or terminal reward) of this node
        up through the tree for all nodes along the path from this node to the root node. Note that the
        value estimate (val_est) is assumed to be from the perspective of the original player
        i.e. self.p_turn.

        val_est (or terminal reward) is added to each node, n_visited is incrimented and virtual_loss is
        decrimented for each node along the path from root to leaf (this node).
        """
        if self.is_terminal:  # If the node is a terminal node, then use the definitive outcome reward
            reward = self.terminal_reward
        else:  # Otherwise, use the value estimate produce by the model instead
            reward = val_est

        node = self
        while node is not None:
            node.n_visits += 1
            node.virtual_loss -= 1
            node.value_sum += reward
            node = node.parent  # Move up to the next node along the path to the root


from collections import defaultdict

def monte_carlo_tree_search(state: str, model, prior_heuristic: Callable, batch_size: int = 32,
                            n_iters: int = 200, **kwargs) -> Tuple[int, float, np.ndarray]:
    """
    Performs Monte Carlo Tree Search (MCTS) for n iterations where model is used as a state value approximator
    to perform bootstrapping.

    :param state: A FEN string denoting the current game state.
    :param model: A value function approximator that produces board value estimates from the perspective of
        the players whose turn it is with model(state_batch) where state_batch is a list of FEN strings.
    :param prior_heuristic: A function that takes in a FEN string state representation and outputs a value
        estimate (should be very fast to compute) for the player whose turn it is next.
    :param batch_size: The size of the state_batch (a list of strings) fed to model.
    :param n_iters: The number of iterations to run (i.e. nodes to expand) when running MCTS.
    :return:
        - best_action (int): The best action found in the search process.
        - state_value (float): The estimated value of the input starting state.
        - action_values (np.ndarray): The estimated value of each possiable next action.
    """
    cache = {} # Cache the values output from the model, if we send it the same state 2x re-use prior values
    root = Node(state=state, parent=None)  # Create a search tree root node
    if root.is_terminal:  # If the input state is a terminal state, no searching required, board value known
        return None, root.terminal_reward, None
    else:  # Expand the root node to get first generation child nodes
        root.expand_legal_moves(prior_heuristic)

    for i in range(n_iters // batch_size + 1):  # Run a batched MC process for batched model forward passes
        leaf_nodes = []  # Record leaf nodes to be passed to the model for batched evaluation

        # 1). Select leaf nodes for expansion in batches for batched evaluation through model(state_batch)
        for k in range(batch_size): # Make batch_size leaf node selections
            if root.unvisited_leaf_nodes == 0:  # If all nodes have been explored, stop iterating, there are
                break  # no further unexplored leaf nodes left, all leaf nodes are terminal nodes

            node = root # All paths in the tree begin at the root node

            while node.is_expanded and not node.is_terminal:  # Traverse down the tree through the nodes that
                # have already been visited before (expanded) and stop when we reach a node that is either
                # terminal (visited but no children) or unvisited (unexpanded). Select the next action
                # (child node) using a prior-informed upper confidence bound for trees selection algorithm
                node = node.argmax_PUCT_child()  # Select a child node using Upper Confidence Bounds

            leaf_node = node  # Change alias now that we've reached the end of the node path
            # Use virtual loss to discourage other iters from selecting a similar path and decriment the
            # unvisited_leaf_nodes counters along this path to prevent duplicate selections
            node.incriment_virtual_loss()
            leaf_nodes.append(leaf_node) # Add this leaf node to our selection of leaf_nodes

        # 2). Collect the states from each leaf node and evaluate with 1 batch through the model
        state_batch, leaf_node_idx = [], []
        for i, leaf_node in enumerate(leaf_nodes):
            # We don't need to run terminated states or ones already seen before through the model, skip them
            if not leaf_node.is_terminal and leaf_node.state_ not in cache:
                state_batch.append(leaf_node.state)
                leaf_node_idx.append(i)
                cache[leaf_node.state_] = None # Add a placeholder in the cache, this will prevent us from
                # running duplicative states through the model for leaf nodes within the current leaf batch

        value_batch = model(state_batch).cpu().tolist()  # Feed all states in to utilize GPU parallelism
        for idx, val_est in zip(leaf_node_idx, value_batch):  # Update the cache with the new model outputs
            cache[leaf_nodes[idx].state_] = val_est

        # 3). Perform backup operations to propagate the new value estimates upwards in the tree
        for i, leaf_node in enumerate(leaf_nodes):
            if leaf_node.is_expanded:  # Skip over any leaf node that is a repeat within the batch i.e. only
                continue  # every expand a leaf node 1x even if selected more than once in a batch

            if leaf_node.is_terminal:  # Then the value of the game state is know for certain, no est needed
                val_est = leaf_node.terminal_reward
            else:  # Otherwise we will rely on the model to approximate the value of the state
                # Flip sign if evaluated from opposing side, all rewards should be from the side of root
                val_est = cache[leaf_node.state_] * (1 if leaf_node.node_turn is leaf_node.p_turn else -1)

            # Add the estimated value to all nodes along the path from leaf to root and decriment virtual loss
            leaf_node.backup(val_est)
            leaf_node.expand_legal_moves(prior_heuristic)  # Create child nodes (if non-terminal)

    # Now that we have populated the MC tree, identify the best action and the estimated state value
    action_values = np.array([node.Q() for node in root.children])
    best_action, state_value = action_values.argmax(), action_values.max()
    return best_action, state_value, action_values, root


#### TODO: DEBUG TESTING ####
if __name__ == "__main__":
    # state = 'r1bqkb1r/pppp1ppp/5n2/4n3/P1B1P3/8/1PPP1PPP/RNB1K1NR b KQkq - 0 5'
    state = 'rnb1k1nr/pppp1ppp/4p3/P1b5/7q/5N1P/2PPPPP1/RNBQKB1R b KQkq - 2 5'
    board = chess.Board(state)
    dummy_model = lambda x: torch.rand(len(x)) * 2 - 1 # TEMP sample model, fill in for a value approximator

    ## Test the Naive Search algo
    # best_action, state_value, action_values = naive_search(state, dummy_model)
    # print("best_action", best_action)
    # print("state_value", state_value)
    # print("action_values", action_values)

    # Test the minimax search algorithm
    # action_values, nodes_evaluated = _minimax_search(state, dummy_model, {}, max_depth=3)
    # ## TODO: Something is wrong here with the second state example, there shoulnd't be so many 1s in the
    # # returns results, the winning set of moves isn't that broad
    # print(action_values)
    # print(nodes_evaluated)


    # best_action, state_value, action_values  = minimax_search(state, dummy_model, gamma=1, horizon=3)
    # print("best_action", best_action)
    # print("state_value", state_value)
    # print("action_values", action_values)

    ## Test Monte Carlo Tree Search
    best_action, state_value, action_values, root = monte_carlo_tree_search(state, dummy_model,
                                                                            material_heuristic, 32, 500)
    print("best_action", best_action)
    print("state_value", state_value)
    print("action_values", action_values)

    legal_moves = list(board.legal_moves)
    for i, x in enumerate(action_values):
        if x == 1:
            print(legal_moves[i])


    def max_depth(node, d=0):
        if not node.children: # Recursion base-case
            return 1
        else:
            return max([max_depth(child) + 1 for child in node.children])

    # print("max_depth(root)", max_depth(root))


    ## TODO:
    ## Figure out what is wrong with the minimax search, why is it giving me so many 1s?

