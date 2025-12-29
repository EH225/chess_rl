"""
This module builds off the classes contained in core/base_components and defines a class instance for a linear
deep Q-network and a CNN-based deep Q-network that replicates the network architecture of the Google DeepMind
model published in Nature.
"""
from __future__ import annotations
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import numpy as np
import torch, chess, time
from utils.chess_env import ChessEnv, relative_material_diff
from scipy.special import softmax
from typing import Tuple, List, Callable, Optional


########################
### Helper Functions ###
########################
# TODO: Section marker

def count_total_nodes(node) -> int:
    """
    Returns the total number of nodes in the subtree rooted at node.
    """
    total = 1  # Count this node itself as 1
    for child in node.children:
        total += count_total_nodes(child)
    return total


def count_leaf_nodes(node) -> int:
    """
    Returns the total number of leaf nodes in the subtree rooted at node.
    """
    if len(node.children) == 0:  # Recursion base-case
        return 1
    else:
        total = 0
        for child in node.children:
            total += count_leaf_nodes(child)
        return total


def max_depth(node):
    """
    Returns the max depth of the subtree rooted at node where the depth of the root node is set to 0.
    """
    if not node.children:  # Recursion base-case
        return 0
    else:
        return max([max_depth(child) + 1 for child in node.children])


def material_heuristic(state: str) -> float:
    """
    Computes a heuristic evaluation of state value based on net material from the perspective of the player
    whose turn is next according to the FEN state encoding.

    This is computed as the net material difference normalized to be [-1, 1] where each piece is worth:
        pawn (1), knight (3), bishop (3), rook (5), queen (9), king (0)

    The net material sum is divided by 39 since 1*8 + 3*2 + 3*2 + 5*2 + 9*1 = 39 which is close to the max
    possible material difference i.e. one side has only a king, the other has all starting pieces. Values
    are also clipped in case a players obtains additional high-value pieces e.g. queens via pawn promotion.

    :param state: A FEN string denoting the current game state.
    :return: A float value representing the approximate value of the board for the player to move next.
    """
    piece_values = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}
    board = chess.Board(state)  # Convert to a chess.Board object to access the piece map
    net_material = 0
    for p in board.piece_map().values():
        piece_val = piece_values[p.symbol().lower()]  # Get the absolute value of each piece
        if board.turn is True and p.symbol().islower():  # For white, lower-case pieces are foes
            piece_val *= (-1)
        elif board.turn is False and p.symbol().isupper():  # For black, upper-case pieces are foes
            piece_val *= (-1)
        net_material += piece_val

    # 39 is the largest net material difference we expect, but clip at [-1, +1] to avoid outliers
    return np.clip(net_material / 39, -1, 1)


#########################
### Naive Search Algo ###
#########################
# TODO: Section marker

def naive_search(state: str, model, batch_size: int = 64, gamma: float = 1.0, **kwargs
                 ) -> Tuple[int, float, np.ndarray, Tuple[int]]:
    """
    Performs a naive search operation where the next action selected it the one which results in the highest
    estimated next state value according to the model. This function performs a depth 1 forward search.

    :param state: A FEN string denoting the current game state.
    :param model: A value function approximator that produces board value estimates from the perspective of
        the players whose turn it is with model(state_batch) where state_batch is a list of FEN strings.
    :param batch_size: The size of the state_batch (a list of strings) fed to model.
    :param gamma: The temporal discount factor to apply to distant rewards. For chess this is usually 1.0.
    :return:
        - best_action (int): The best action found in the search process.
        - state_value (float): The estimated value of the input starting state.
        - action_values (np.ndarray): The estimated value of each possible next action.
        - info (Tuple[int]): The total number of nodes evaluated, the max depth of the search tree and how
            many terminal game state nodes were visited.
    """
    env = ChessEnv(initial_state=state)  # Instantiate the current game state
    if env.ep_ended:  # If the episode has already ended, then there is no searching over action to be done
        # The value is always computed from the perspective of the player who is to move next in the state,
        # if the game has ended in a checkmate, then the last move from the opponent achieved it and the
        # player to move next has lost, return a value of -1 and 9999 for best_action as a placeholder
        value = -1 if env.board.is_checkmate() else 0
        return 9999, value, np.zeros(0), (1, 0, 1)

    # Else: Perform a search over possible next actions in the env from the current starting state
    action_values = []  # Record a value estimate for each possible next action from the starting state
    state_batches = []  # Record the FEN string encodings of possible next states that we will feed through
    # the model in batches to get out value estimates for each
    state_batches_idx = []  # Record the indices of the actions each state in state_batches corresponds to,
    # we will only run states that are non-terminal through the model to minimize computations i.e. if the
    # state is terminal, no value estimate is required, the outcome is given directly from the env
    terminal_nodes = 0  # Count how many of the nodes reached were terminal

    for i in range(env.action_space.n):  # Consider all possible legal moves
        new_state, reward, terminated, truncated = env.step(i)  # Make the candidate move
        action_values.append(reward)  # Record the immediate reward of this action
        if not terminated:  # Check if the game hasn't reached a terminal state, if so then bootstrap the
            # value, this includes a truncated state from which we could continue to play additional moves
            # estimation thereafter using the value approximator (model)
            if len(state_batches) == 0 or len(state_batches[-1]) == batch_size:
                state_batches.append([new_state, ])  # Start a new list for this next state obs
            else:  # Otherwise, append to the existing batch so that each is batch_size or less
                state_batches[-1].append(new_state)
            state_batches_idx.append(i)  # Make note of the index of action_values to add to later
        else:
            terminal_nodes += 1  # Count how many nodes visited are terminal overall
        env.rev_step()  # Backtrack so that we can try a different move instead

    # Compute the bootstrap value estimates in batches for the next possible states after each action
    value_estimates = []
    for state_batch in state_batches:  # Compute the value estimates in batches to minimize runtime
        # Disable grad-tracking, not needed since no gradient step being taken, use bfloat16 dtypes
        with torch.no_grad(), torch.autocast(device_type=model.device, dtype=torch.bfloat16):
            v_est = model(state_batch).cpu().reshape(-1).tolist()
        value_estimates.extend(v_est)  # Aggregate the state value estimates into 1 linear list

    for idx, val in zip(state_batches_idx, value_estimates):  # Add the bootstrapped value estimates for s'
        # gamma is applied to discount future rewards and the value estimates need to have their signs
        # reversed since they are computed from the perspective of the other player i.e. the one who plays
        # from s', chess is a zero-sum game so the value to one is the negative of the value to the other
        action_values[idx] += gamma * val * (-1)

    # Determine the best action and the highest value by computing the argmax and max
    action_values = np.array(action_values)
    best_action, state_value = action_values.argmax(), action_values.max()
    return best_action, state_value, action_values, (1, len(action_values) + 1, terminal_nodes)


######################################
### Minimax Alpha-Beta Search Algo ###
######################################
# TODO: Section marker

class Node_MMS:
    """
    Class object for a search tree node used in Minimax Search (MMS) with alpha-beta pruning.
    """

    def __init__(self, state: str, parent: Optional[Node_MMS] = None, gamma: float = 1.0,
                 reward: float = 0.0):
        """
        Instantiates a MMS tree node.

        :param state: A FEN string denoting the current game state.
        :param parent: A pointer to this node's parent node in the search tree. Will be None for the root.
        :param gamma: The temporal discount factor to apply to distant rewards. For chess this is usually 1.0.
        :param reward: The reward obtained at the parent node for taking the action that results in this node.
        """
        self.state = state  # Record the FEN state encoding of this game state
        self.state_ = " ".join(state.split()[:-1])  # Remove the total move counter from the FEN
        board = chess.Board(state)  # Init a chess board obj for internal ops, skip recording to save memory
        self.parent = parent  # Record a pointer to this node's parent node, will be None if this is the root
        self.gamma = gamma  # Temporal discount factor
        # Make note of whose turn it is (white or black) at the root node. If parent is None, then this is a
        # root node, record from the current game state, otherwise inherit from this from the parent node
        self.root_turn = board.turn if parent is None else parent.root_turn
        self.node_turn = board.turn  # Record the turn value for this game state's next-to-move
        # The reward is generated from the env from the perspective of the prior player to move, so we flip
        # it if the current player is the root_turn player since the prior player was the opponent
        self.reward = reward * (-1 if board.turn == self.root_turn else 1)
        self.maximize = self.root_turn == self.node_turn  # A bool if this node is a maximization node
        # Maintain a list of actions from this state that we have not yet created child nodes for, go in order
        # from smallest to largest when creating new actions i.e. start with action 0, 1, 2... etc.
        self.unexplored_actions = list(range(board.legal_moves.count()))[::-1]
        self.children = []  # Maintain pointers to each child node

        if parent is not None:  # alpha and beta values are inherited from parent nodes, alpha and beta values
            # only travel downwards in the tree, not upwards. Higher up values of alpha and beta are updated
            # when node values are updated
            self.alpha, self.beta = parent.alpha, parent.beta  # Inherit from above
            self.depth = self.parent.depth + 1
        else:  # Otherwise for the root node, init alpha and beta at (-inf, inf)
            self.alpha, self.beta = -float("inf"), float("inf")
            self.depth = 0  # Root nodes have a depth of zero, depth = actions taken from root

        self.is_terminal = board.is_game_over()
        if self.is_terminal:  # If the game is over, then record the value of starting in this state as 0
            self.value = 0  # since there are no further moves that can be made
        else:  # Value keeps track of the best value seen so far for the current node among its children
            self.value = -float("inf") if self.maximize is True else float("inf")
        self.fully_expanded = False  # Track if the node has been fully expanded i.e. update_tree called on it
        self.children_pruned = False  # Set to True if child nodes are pruned early

    def update_tree(self) -> None:
        """
        This method is called when a node has been fully expanded i.e. if it is terminal, at max depth, or
        has had all of its child nodes fully expanded. This method propagates the current nodes information
        up the tree until it reaches the root node. This method is only called 1x per node.
        """
        assert not self.fully_expanded, "update_tree method called more than once on the same node"
        # reward = payoff immediately from taking the action leading from parent -> child
        # value = the best value the player playing at node can get starting at node and playing thereafter
        self.fully_expanded = True  # Once this method is called, the node is fully expanded
        val = self.reward + self.gamma * self.value  # If terminal, then self.value == 0
        node = self.parent
        while node is not None:  # Iterate through nodes from leaf to root and update as we go
            if node.maximize:  # Node is a maximizer turn
                node.value = max(node.value, val)  # Will always select the max among the child node values
                node.alpha = max(node.alpha, val)  # Track the best obtainable value
            else:  # Node is a minimizer turn instead
                node.value = min(node.value, val)  # Will always select the min among the child node values
                node.beta = min(node.beta, val)  # Track the best obtainable value

            # Update val for next iteration at this node's parent node (if any)
            val = node.reward + node.gamma * node.value
            node = node.parent  # Update the node pointer for next iteration, move to the parent of this node

    def __str__(self) -> str:
        return self.state

    def __repr__(self) -> str:
        return self.state

def minimax_search(state: str, model, gamma: float = 1.0, batch_size: int = 64, horizon: int = 3, **kwargs
                   ) -> Tuple[int, float, np.ndarray, Tuple[int]]:
    """
    Performs minimax-search with alpha-beta pruning up to a particular search horizon after which the model
    is used as a value approximator for each state to perform bootstrapping.

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
    from the minimizer of 3. Then when evaluating the second child node, the maximizer will be looking for
    something that is >= 3. If during the expansion of that node, we find that a value of 2 is an option for
    the minimizer at that node, that will be an upper bound for anything the minimizer chooses for that node.
    Therefore, we can stop expanding branches along the second child node since we now know the max value of
    that node will be 2 and the maximizer will always prefer the first child node over it instead.

    :param state: A FEN string denoting the current game state.
    :param model: A value function approximator that produces board value estimates from the perspective of
        the players whose turn it is with model(state_batch) where state_batch is a list of FEN strings.
    :param gamma: The temporal discount factor to apply to distant rewards. For chess this is usually 1.0.
    :param batch_size: The size of the state_batch (a list of strings) fed to model.
    :param hoirzon: The max depth of the search tree at which point the model is used to estimate values
        instead.
    :return:
        - best_action (int): The best action found in the search process.
        - state_value (float): The estimated value of the input starting state.
        - action_values (np.ndarray): The estimated value of each possible next action.
        - info (Tuple[int]): The total number of nodes evaluated, the max depth of the search tree and how
            many terminal game state nodes were visited.
    """
    board = chess.Board(state)  # Compute the reward that would have been generated on the move prior to
    # reaching the current game state i.e. +1 reward for the prior player if the board is now a checkmate
    # and 0 otherwise (i.e. for intermediate moves or any kind of draw condition)
    reward = 1 if board.is_checkmate() else 0
    root = Node_MMS(state=state, parent=None, reward=reward)  # Reward flipping done internally
    if root.is_terminal:  # If a terminal state, no searching to be done, use 9999 as the best action
        # placeholder since an integer is expected to be returned
        return 9999, root.reward, np.zeros(0), (1, 0, 1)
    elif horizon == 0:  # If the horizon is zero, then use the model to evaluate and return that value
        with torch.no_grad(), torch.autocast(device_type=model.device, dtype=torch.bfloat16):
            vale_est = model([state]).cpu().reshape(-1).tolist()[0]
        return 9999, vale_est, np.zeros(0), (1, 0, 0)

    # Run minimax search with alpha-beta pruning using DFS with batched leaf-evaluation
    cache = {}  # Cache model evaluations so that they don't need to be re-run
    eval_batch = []  # Collect nodes for batched model evaluation
    terminal_nodes = 0  # Count how many of the nodes reached were terminal

    node_stack = [root]  # Maintain a stack of nodes to run expansions of or to revisit
    while node_stack:  # Iterate until all searching has been done
        node = node_stack.pop()

        if node.is_terminal:  # No further searching possible, update the parent node with the terminal value
            terminal_nodes += 1  # Count how many nodes visited are terminal overall
            node.update_tree()

        elif node.depth == horizon:  # No further searching, use the model to estimate the node's value
            node_value = cache.get(node.state_, None)  # Check if already pre-computed
            if node_value is None:  # If not pre-computed, add to the next evaluation batch for delayed update
                eval_batch.append(node)
            else:  # If the state's model eval is pre-cached and available, then update the parent node now
                node.value = node_value
                node.update_tree()

        elif node.unexplored_actions:  # If there are still unexplored actions, explore all child nodes
            # Check for alpha-beta pruning conditions. If met, then no further searching is needed along this
            # branch, stop further child node evaluation and update the parent of this node

            if node.maximize is True and node.alpha > node.beta:  # Must be a strict inequality so that
                # the best move at the root can be correctly identified
                # node.value is the game score the maximizer can guarantee from pathways beyond this current
                # node. The maximizer will select the highest value among the children. Therefore, the current
                # value of node.value is a lower bound as to what the maximizer will select. beta tells us
                # that the minimizer can select a value of beta somewhere else at the same level as this node
                # therefore the minimizer 1 level up will never pick this node since the value of this node
                # will be at least node.value which is worse than what the minimizer can get elsewhere.
                node.fully_expanded = True  # Once we hit the pruning condition, stop exploring, set this node
                # as fully-expanded, no need to call update since leaf nodes already updated
                node.children_pruned = True  # Set a flag if pruning was done

            elif node.maximize is False and node.beta < node.alpha:  # Must be a strict inequality so that
                # the best move at the root can be correctly identified
                # node.value is the game score the minimizer can guarantee from pathways beyond this current
                # node. The minimizer will select the lowest value among the children. Therefore, the current
                # value of node.value is an upper bound as to what the minimizer will select. alpha tells us
                # that the maximizer can select a value of alpha somewhere else at the same level as this node
                # therefore the maximizer 1 level up will never pick this node since the value of this node
                # will be at most node.value which is worse than what the maximizer can get elsewhere.
                node.fully_expanded = True  # Once we hit the pruning condition, stop exploring, set this node
                # as fully-expanded, no need to call update since leaf nodes already updated
                node.children_pruned = True  # Set a flag if pruning was done

            else:  # If we're not pruning the further children of this node, explore the next one
                node_stack.append(node)  # Append again since we will want to come back to evaluate if this
                # node has additional children in the future, and if not, then to update its parent with the
                # value of the node once all child exploration has finished

                action = node.unexplored_actions.pop()  # Get the next action to consider (an int)
                env = ChessEnv(initial_state=node.state)  # Create a ChessEnv to model actions
                next_state, reward, terminated, truncated = env.step(action)
                node_stack.append(Node_MMS(state=next_state, parent=node, gamma=gamma, reward=reward))
                node.children.append(node_stack[-1])  # Add a pointer from the parent node to the child node

        else:  # The node is non-terminal, not at max depth, but also doesn't have any further unexplored
            # child nodes, so nothing to do here no update_tree() call needed, each leaf node triggers it
            node.fully_expanded = True

        # After each node exploration / expansion operation, check if the eval_batch is full and should be
        # run through the model or if there are no further nodes on the stack i.e. run the last set of evals
        if len(eval_batch) == batch_size or len(node_stack) == 0:
            # If the eval_batch has reached batch_size or if the node_stack for further exploration is
            # depleted, then evaluate the nodes contained in eval_batch and update the tree accordingly
            state_batch = [node.state for node in eval_batch]  # Extract a list of FEN state encodings (str)
            with torch.no_grad(), torch.autocast(device_type=model.device, dtype=torch.bfloat16):
                value_batch = model(state_batch).cpu().reshape(-1).tolist()

            for node, value in zip(eval_batch, value_batch):  # Update the tree with these value estimates
                # Record the value approximation of this node and deal with sign flipping to make the value
                # estimate from the perspective of the root node (a maximizer node)
                value *= (1 if node.maximize else -1) # Flip the sign if the opponent is to move next
                cache[node.state_] = value  # Add this value to the cache once computed
                node.value = value # Set as the node's value once obtained
                node.update_tree()  # Once populated, backup the update throughout the tree

            eval_batch = []  # Once this batch is finished, clear out the buffer for the next batch of nodes

    # Once finished with the search process, extract the best action, overall state est, and action values
    action_values = np.array([child.reward + child.value * gamma for child in root.children])
    best_action, state_value = action_values.argmax(), action_values.max()
    return best_action, state_value, action_values, (count_total_nodes(root), max_depth(root), terminal_nodes)


####################################
### Monte Carlo Tree Search Algo ###
####################################
# TODO: Section marker

class Node_MCTS:
    """
    Class object for a search tree node used in Monte Carlo Tree Search (MCTS).
    """

    def __init__(self, state: str, parent: Optional[Node_MCTS]):
        """
        Instantiates an MCTS tree node.

        :param state: A FEN string denoting the current game state.
        :param parent: A pointer to this node's parent node in the search tree. Will be None for the root.
        """
        self.state = state  # Record the FEN state encoding of this game state
        self.state_ = " ".join(state.split()[:-1])  # Remove the total move counter from the FEN
        board = chess.Board(state)  # Init a chess board obj for internal ops, skip recording to save memory
        self.parent = parent  # Record a pointer to this node's parent node, will be None if this is the root
        # Make note of whose turn it is (white or black) at the root node. If parent is None, then this is a
        # root node, record from the current game state, otherwise inherit from this from the parent node
        self.root_turn = board.turn if parent is None else parent.root_turn
        self.node_turn = board.turn  # Record the turn value for this game state's next-to-move
        self.value_sum = 0  # Record the sum of all value updates to this node, starts at 0
        self.n_visits = 1 if parent is None else 0  # Record the total node visits, root begins with 1
        self.is_expanded = False  # Bool flag for if this node has been expanded yet, beings at False

        # Bool flag indicating whether this is a terminal game state node
        self.is_terminal = board.is_game_over()
        if self.is_terminal:  # If the game is over, record the terminal reward value from the view of root
            # If the game ends in checkmate and root_turn is next to move, then root lost
            if board.is_checkmate():
                self.terminal_reward = -1 if self.node_turn is self.root_turn else 1
            else:  # For all other types of end games (stalemates, insufficient materials etc.)
                self.terminal_reward = 0
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
        Returns the average of all rewards backpropagated though this node i.e. value_sum / n_visits. If the
        node has never been visited before, then the returned value is 0 (a neutral value between -1 and +1,
        the two max values we could have).
        """
        return self.value_sum / self.n_visits if self.n_visits > 0 else 0

    def argmax_PUCT_child(self) -> Optional[Node_MCTS]:
        """
        Uses Predictor Upper Confidence Bound for Trees (PUCT) to select a child action with a virtual loss
        to discourage making the same selections multiple times.

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
        if not self.children or self.unvisited_leaf_nodes == 0:  # Return None if there are no child nodes
            return None  # yet unexplored down this route

        # Look for the max PUCT value among all child nodes with unvisited leaf nodes
        argmax_node, max_val = None, -float("inf")
        for child in self.children:
            if child.unvisited_leaf_nodes == 0:  # Skip if everything has already been explored
                continue
            else:
                n_eff = child.n_visits + child.virtual_loss  # virtual_loss discourages repeat selections
                val = child.value_sum / n_eff if n_eff > 0 else 0  # Avoid division by 0 issues
                # At this node, we will select the the child whose value is most appealing to this node's
                # perspective, all value_sums are from the root's perspective so flip the signs if this node
                # is played by the opponent who would want to minimize the return of the root node
                val *= (1 if self.node_turn == self.root_turn else -1)
                c_puct = self.c_init + np.log(n_eff + self.c_base + 1) - np.log(self.c_base)
                # child.prior is recorded from the perspective of the this node, the parent node so no sign
                # flip is needed, they're also normalized using softmax so it's non-trivial to flip
                val += c_puct * child.prior * np.sqrt(self.n_visits) / (1 + n_eff)
                if val > max_val:  # Update the argmax node seen so far, select the highest value node among
                    # all child nodes from this node's turn perspective
                    argmax_node, max_val = child, val

        return argmax_node

    def expand_legal_moves(self, prior_heuristic: Callable = None) -> None:
        """
        This method expands the current node i.e. adds unexplored child nodes to self.children if applicable
        (i.e. if this node is non-terminal) and sets self.is_expanded to True. The child nodes created have
        their prior values set by softmax(prior_heuristic(child.state)) if provided, otherwise a uniform prior
        of 1 / num_actions is used where num_actions == n_children. The prior values are configured to sum to
        1 across all actions

        :param prior_heuristic: A heuristic function for computing a prior value for each child node added.
            Should be a function that takes the FEN state (str) as its first argument and returns the value
            estimate for the player whose move is next.
        :return: None, Node objects are added as children to the current node.
        """
        assert not self.is_expanded, "This node has already been expanded"
        unvisited_leaf_nodes_chg = -1  # Record how many new unexplored leaf nodes are created, this node
        # goes from being an unvisted / unexpanded node to expanded so we minus 1 to start with
        if not self.is_terminal:  # If not terminal, then there are additional child nodes we can add
            board = chess.Board(self.state)  # Init a chess board object for internal operations
            legal_moves = list(board.legal_moves)
            uniform_prior = 1 / len(legal_moves)  # Sums to 1 across all actions by design
            for move in legal_moves:  # Add a child node for each legal move starting here
                board.push(move)  # Make this move on the board to get the next resulting game state
                child = Node_MCTS(state=board.fen(), parent=self)
                if prior_heuristic is None:  # If no prior_heuristic provided, then set to the unif prior 1/n
                    child.prior = uniform_prior
                else:  # If a prior_heuristic is given, then use it to generate a prior for each child node
                    # The prior_heuristic will be evaluated from the perspective of the child i.e. the player
                    # to go next in the child node state, flip the sign since it will only be accessed by
                    # the parent node in making an expansion selection, will be passed through softmax
                    child.prior = prior_heuristic(child.state) * (-1)
                self.children.append(child)
                board.pop()  # Backtrack to visit the next legal move
                unvisited_leaf_nodes_chg += 1  # Record another unexplored child node added

            # Once we've added all the prior values, normalize across them using softmax to ensure that they
            # are all positive and sum to 1 across all actions (children)
            normed_priors = softmax([child.prior for child in self.children])
            if self.parent is None:  # Add dirichlet noise at the parent node to the child prior values
                rng = np.random.default_rng()
                noise = rng.dirichlet([0.3 for i in range(len(self.children))], size=1).reshape(-1)
                normed_priors = (1 - 0.25) * normed_priors + (0.25) * noise
            assert np.abs(normed_priors.sum() - 1.0) < 1e-6, "normed_priors do not sum to 1.0"

            for i, prior_val in enumerate(normed_priors):  # Update values after applying softmax collectively
                self.children[i].prior = prior_val

            node = self  # Back propagate the update for unvisited_leaf_nodes to the root node
            while node is not None:
                node.unvisited_leaf_nodes += unvisited_leaf_nodes_chg
                node = node.parent

        self.is_expanded = True  # Set this flag to true now that this node has been expanded with children

    def incriment_virtual_loss(self) -> None:
        """
        Starting at this node, this method increments the virtual_loss counters for all nodes along the path
        from this node to the root node and also decrements the unvisited_leaf_nodes counters to prevent
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
        i.e. self.root_turn.

        val_est (or terminal reward) is added to each node, n_visited is incremented and virtual_loss is
        decremented for each node along the path from root to leaf (this node).
        """
        if self.is_terminal:  # No estimate needed, definitive outcome from game env
            msg = "val_est should be equal to terminal_reward for a terminal node but got: "
            msg += f"val_est={val_est} and self.terminal_reward={self.terminal_reward}"
            assert val_est == self.terminal_reward, msg

        node = self
        while node is not None:
            node.n_visits += 1
            node.virtual_loss -= 1
            node.value_sum += val_est
            node = node.parent  # Move up to the next node along the path to the root


def monte_carlo_tree_search(state: str, model, batch_size: int = 32, n_iters: int = 200,
                            prior_heuristic: Callable = relative_material_diff,
                            **kwargs) -> Tuple[int, float, np.ndarray, Tuple[int]]:
    """
    Performs Monte Carlo Tree Search (MCTS) for n iterations where model is used as a state value approximator
    to perform bootstrapping.

    :param state: A FEN string denoting the current game state.
    :param model: A value function approximator that produces board value estimates from the perspective of
        the players whose turn it is with model(state_batch) where state_batch is a list of FEN strings.
    :param batch_size: The size of the state_batch (a list of strings) fed to model.
    :param prior_heuristic: A function that takes in a FEN string state representation and outputs a value
        estimate (should be very fast to compute) for the player whose turn it is next.
    :param n_iters: The number of iterations to run (i.e. nodes to expand) when running MCTS.
    :return:
        - best_action (int): The best action found in the search process.
        - state_value (float): The estimated value of the input starting state.
        - action_values (np.ndarray): The estimated value of each possible next action.
        - info (Tuple[int]): The total number of nodes evaluated, the max depth of the search tree and how
            many terminal game state nodes were visited.
    """
    cache = {}  # Cache the values output from the model, if we send it the same state 2x re-use prior values
    terminal_nodes = 0  # Count how many of the nodes reached were terminal
    nodes_expanded = 0 # Count how many total nodes are expanded during MCTS
    root = Node_MCTS(state=state, parent=None)  # Create a search tree root node
    if root.is_terminal:  # If the input state is a terminal state, no searching required, board value known
        return 9999, root.terminal_reward, np.zeros(0), (1, 0, 1)
    else:  # Expand the root node to get first generation child nodes
        root.expand_legal_moves(prior_heuristic)

    for i in range(n_iters // batch_size + 1):  # Run a batched MC process for batched model forward passes
        leaf_nodes = []  # Record leaf nodes to be passed to the model for batched evaluation

        if root.unvisited_leaf_nodes == 0:  # If all nodes have been explored, stop iterating, there are
            break  # no further unexplored leaf nodes left, all leaf nodes are terminal nodes, stop searching

        # 1). Select leaf nodes for expansion in batches for batched evaluation through model(state_batch)
        for k in range(batch_size):  # Make batch_size leaf node selections
            if root.unvisited_leaf_nodes == 0:  # If all nodes have been explored, stop iterating, there are
                break  # no further unexplored leaf nodes left, all leaf nodes are terminal nodes

            node = root  # All paths in the tree begin at the root node

            while node.is_expanded and not node.is_terminal:  # Traverse down the tree through the nodes that
                # have already been visited before (expanded) and stop when we reach a node that is either
                # terminal (visited but no children) or unvisited (unexpanded). Select the next action
                # (child node) using a prior-informed upper confidence bound for trees selection algorithm
                node = node.argmax_PUCT_child()  # Select a child node using Upper Confidence Bounds

            leaf_node = node  # Change alias now that we've reached the end of the node path
            # Use virtual loss to discourage other iters from selecting a similar path and decriment the
            # unvisited_leaf_nodes counters along this path to prevent duplicate selections
            node.incriment_virtual_loss()
            leaf_nodes.append(leaf_node)  # Add this leaf node to our selection of leaf_nodes
            if leaf_node.is_terminal:
                terminal_nodes += 1  # Count how many nodes visited are terminal overall
            nodes_expanded += 1

        # 2). Collect the states from each leaf node and evaluate with 1 batch through the model
        state_batch, leaf_node_idx = [], []
        for i, leaf_node in enumerate(leaf_nodes):
            # We don't need to run terminated states or ones already seen before through the model, skip them
            if not leaf_node.is_terminal and leaf_node.state_ not in cache:
                state_batch.append(leaf_node.state)
                leaf_node_idx.append(i)
                cache[leaf_node.state_] = None  # Add a placeholder in the cache, this will prevent us from
                # running duplicative states through the model for leaf nodes within the current leaf batch

        with torch.no_grad(), torch.autocast(device_type=model.device, dtype=torch.bfloat16):
            value_batch = model(state_batch).cpu().reshape(-1).tolist()  # Run in parallel on the GPU
        for idx, val_est in zip(leaf_node_idx, value_batch):  # Update the cache with the new model outputs
            cache[leaf_nodes[idx].state_] = val_est

        # 3). Perform backup operations to propagate the new value estimates upwards in the tree
        for i, leaf_node in enumerate(leaf_nodes):
            # We only ever select and expand a leaf node 1x, assert that this is leaf hasn't already been
            # reached and expanded before, if so then raise an assertion error
            assert not leaf_node.is_expanded, "leaf node already expanded"

            if leaf_node.is_terminal:  # Then the value of the game state is know for certain, no est needed
                val_est = leaf_node.terminal_reward
            else:  # Otherwise we will rely on the model to approximate the value of the state
                # Flip sign if evaluated from opposing side, all rewards should be from the side of root
                val_est = cache[leaf_node.state_] * (1 if leaf_node.node_turn == leaf_node.root_turn else -1)

            # Add the estimated value to all nodes along the path from leaf to root and decrement virtual loss
            leaf_node.backup(val_est)
            leaf_node.expand_legal_moves(prior_heuristic)  # Create child nodes (if non-terminal)

    # Now that we have populated the MC tree, identify the best action and the estimated state value
    action_values = np.array([node.Q() for node in root.children])
    best_action, state_value = action_values.argmax(), action_values.max()
    return best_action, state_value, action_values, (nodes_expanded, max_depth(root), terminal_nodes)


#### TODO: DEBUG TESTING ####
if __name__ == "__main__":
    import time

    state = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'  # Default starting state
    state = 'r1bqkb1r/pppp1ppp/5n2/4n3/P1B1P3/8/1PPP1PPP/RNB1K1NR b KQkq - 0 5'  # Regular board
    state = 'rnb1k1nr/pppp1ppp/4p3/P1b5/7q/5N1P/2PPPPP1/RNBQKB1R b KQkq - 2 5'  # 1 move away from checkmate
    state = 'rnb1k1r1/pPppnppp/4p3/2b5/7q/5N1P/2PPPPP1/RNBQKB1R w KQq - 1 8'  # Pawn promotion
    state = 'rnb1k1nr/pppp1ppp/4p3/P1b5/8/5N1P/2PPPqP1/RNBQKB1R w KQkq - 0 6'  # Checkmate
    board = chess.Board(state)
    dummy_model = lambda x: torch.rand(len(x)) * 2 - 1  # TEMP sample model, fill in for a value approximator


    ## Test the Naive Search algo
    # best_action, state_value, action_values = naive_search(state, dummy_model)
    # print("best_action", best_action)
    # print("state_value", state_value)
    # print("action_values", action_values)

    ## Test the Minimax Search Algo
    # best_action, state_value, action_values = minimax_search(state, dummy_model, gamma=1, horizon=4)
    # print("best_action", best_action) # Should pick 13
    # print("state_value", state_value)
    # print("action_values", action_values) # Should be 46 actions from the initial board state
    # print(list(board.legal_moves)[best_action])

    # Test Monte Carlo Tree Search
    # start_time = time.time()
    # best_action, state_value, action_values = monte_carlo_tree_search(state, dummy_model, 32, 500,
    #                                                                   material_heuristic)
    # print(f"Runtime: {time.time() - start_time:.2f}s")
    # print("best_action", best_action)
    # print("state_value", state_value)
    # print("action_values", action_values)

    # board = chess.Board(state)
    # legal_moves = list(board.legal_moves)
    # for i, x in enumerate(action_values):
    #     if x == 1:
    #         print(legal_moves[i])
