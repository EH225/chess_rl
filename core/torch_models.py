"""
This module builds off the classes contained in core.base_components.py and defines a class instance for 3
different value-estimator modeling approachs:
    1. A multi-layer perceptron (MLP) neural network
    2. A CNN-based neural network
    3. A transformer-based model
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import numpy as np
import torch, chess
import torch.nn as nn
from core.base_components import DVN
from utils.general import compute_img_out_dim
from typing import Tuple, List


####################################
### MLP Value-Network Definition ###
####################################
# TODO: Section marker

class ResBlockMLP(nn.Module):
    """
    Residual block sub-unit of MLP network model class.
    """
    def __init__(self, size: int):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(x + h)


class MLP(DVN):
    """
    Implementation of a multi-layer perceptron (MLP) / fully-connected neural network (FCNN) model for board
    value estimation.
    """
    def initialize_model(self) -> None:
        """
        Initializes the required value network model as a fully-connected neural network. This network
        architecture follows that of the TD-λ evaluator.

        The input to this network will be a batch of FEN string state representation of the current game
        state. self.v_network will be the learned value-function approximator. This method also instantiates
        an optimizer for training.
        """
        input_shape = 8 * 8 * 6 + 5 # 8 rows, 8 cols, 6 piece types, -1, 0, 1 values denoting a piece as
        # friendly or foe or if the cell is empty for a total size of 384 input features + 4 castling rights
        # and +1 50-move rule counter (draw condition)
        self.v_network = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.LeakyReLU(),
            ResBlockMLP(512),
            ResBlockMLP(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        ) # Use a final Tanh activation function at the end to produce value estimates [-1, +1]

        # Init the optimizer for training, add L2 regularization through the weight_decay parameter
        self.optimizer = torch.optim.Adam(self.v_network.parameters(),
                                          weight_decay=self.config["model"]["wt_decay"])

    def state_to_model_input(self, state_batch: List[str]) -> torch.Tensor:
        """
        Converts an input batch of board states (encoded using FEN as a string) into the expected state
        representations for this model (torch.Tensor) and moves the data to the same device as this model.

        This MLP model operates on board representations that are [8 x 8 x 6] which record the spatial
        location of each piece and type along each dimension while the value denotes (+1) for friend,
        (-1) for foe, or (0) for an empty square. In the last dimension, index 0 = pawn, index 1 = knight,
        index 2 = bishop, index 3 = rook, index 4 = queen, index 5 = king.

        Locationally, the friendly pieces are always shown at the bottom of the board and foe pieces always
        are shown at the top of the board. Therefore, there is some board flipping depending on the color
        whose move it is next so that the model learns the same chess playing from both perspectives.

        The [8 x 8 x 6] board tensors are flattened and then a few additional info nodes are appended to
        indicate if each side can still king-side castle or queen-side castle and how close the current match
        is to reaching the 50 move draw counter (if no pawn moves or captures).

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, (8 * 8 * 6 + 4 + 1)) = (batch_size, 389)
        """
        sym_to_int = {s:i for i, s in enumerate("pnbrqk")}  # Mapping from symbol e.g. "b" to index e.g. 2
        output = torch.zeros((len(state_batch), 8, 8, 6))  # 8 rows, 8 cols, 6 piece types, encode -1, 0, or 1
        extra_info = torch.zeros((len(state_batch), 5)) # 4 castling rights + 1 repition counter
        for i, state in enumerate(state_batch):
            board = chess.Board(state) # Use the FEN string encoding to create the board
            for cell, piece in board.piece_map().items():  # Iter over the piece dictionary
                r, c = divmod(cell, 8)  # Get the row and column of this piece
                if board.turn:  # By default, White's pieces will be in cells 0, 1, ... etc. and black's will
                    # be in cells ... 63, 64 etc. We want to have friendly pieces always in cells (7, ...)
                    # so if it's black's turn to move, we will leave the rows as-is, otherwise if it's white's
                    # turn to move, then we will horizontally flip the rows so that the model always sees
                    # friendly pieces in the bottom rows and foe pieces in the top rows
                    r = 7 - r # Invert the row value so that friendly pieces are always in the bottom rows
                if board.turn:  # If it is white's move, then upper-case (white) are the player's pieces
                    val = -1 if piece.symbol().islower() else 1
                else: # If it is black's move, then upper-case (white) pieces are the opponent's
                    val = 1 if piece.symbol().islower() else -1
                output[i, r, c, sym_to_int[piece.symbol().lower()]] = val

            # A). Add castling rights on the king and queen side for friendly and foe
            if board.turn is chess.WHITE: # If it is white's turn, then white is friendly and black is foe
                friendly_color, foe_color = chess.WHITE, chess.BLACK
            else: # Otherwise if it is black's turn, then black is friendly and white is foe
                friendly_color, foe_color = chess.BLACK, chess.WHITE
            extra_info[i, 0] = 1 if board.has_kingside_castling_rights(friendly_color) else 0
            extra_info[i, 1] = 1 if board.has_queenside_castling_rights(friendly_color) else 0
            extra_info[i, 2] = 1 if board.has_kingside_castling_rights(foe_color) else 0
            extra_info[i, 3] = 1 if board.has_queenside_castling_rights(foe_color) else 0

            # B). 50 move rule counter - Encodes the number of half-moves since last capture or pawn move
            # when this counter reaches 100, then 50 whole moves have been made and the game ends in a draw
            extra_info[i, 4] = min(board.halfmove_clock, 100) / 100.0 # Scale to be [0, 1]

        output = output.flatten(start_dim=1) # (batch_size, 8, 8, 6) -> (batch_size, 384)
        # Add info about castling rights and how close are to a draw based on the 50-move rule
        output = torch.concatenate([output, extra_info], dim=1)  #  (B, 384) +  (B, 5) =  (B, 389)
        # Move the model_input to the required device so it can be run through the network before returning
        return output.to(self.device)

    def forward(self, state_batch: List[str]) -> torch.Tensor:
        """
        Forward pass through the model which generates a value estimate of the current board position for
        each state observation in the input state_batch i.e. an estimate of the expected reward from the
        current state position.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, ) with the value estimates for each stating position.
        """
        # Convert the input board into the expected state representation and pass it through the network
        return self.v_network(self.state_to_model_input(state_batch))


####################################
### CNN Value-Network Definition ###
####################################
# TODO: Section marker

class ResBlockCNN(nn.Module):
    """
    Residual block sub-unit of CNN network model class.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.conv1(x))
        h = self.conv2(h)
        return self.activation(x + h)


class CNN(DVN):
    """
    Implementation of a convolutional neural network model for board value estimation.
    """
    def initialize_model(self) -> None:
        """
        Initializes the required value network model as a convolutional neural network (CNN). This network
        architecture follows a blend of various other resnet-based CNNs including the one from AlphaZero.

        The input to this network will be a batch of FEN string state representation of the current game
        state. self.v_network will be the learned value-function approximator. This method also instantiates
        an optimizer for training.
        """
        self.v_network = nn.Sequential(
            # Conv2d Block 1
            nn.Conv2d(in_channels=17, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(), # (batch_size, 64, 8, 8)
            ResBlockCNN(64), # (batch_size, 64, 8, 8)

            # Conv2d Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(), # (batch_size, 128, 8, 8)
            ResBlockCNN(128), # (batch_size, 128, 8, 8)

            # Dense fully-connected Block 3
            nn.flatten(), # (batch_size, 128, 8, 8) -> (batch_size, 8192)
            nn.Linear(128 * 8 * 8, 512), # (batch_size, 8192) -> (batch_size, 512)
            nn.LeakyReLU(),
            nn.Linear(512, 128), # (batch_size, 512) -> (batch_size, 128)
            nn.LeakyReLU(),
            nn.Linear(128, 1), # (batch_size, 128) -> (batch_size, 1)
            nn.Tanh()
        ) # Use a final Tanh activation function at the end to produce value estimates [-1, +1]

        # Init the optimizer for training, add L2 regularization through the weight_decay parameter
        self.optimizer = torch.optim.Adam(self.v_network.parameters(),
                                          weight_decay=self.config["model"]["wt_decay"])

    def state_to_model_input(self, state_batch: List[str]) -> torch.Tensor:
        """
        Converts an input batch of board states (encoded using FEN as a string) into the expected state
        representations for this model (torch.Tensor) and moves the data to the same device as this model.

        This CNN model operates on board representations that are [(12 + 4 + 1 + 1 + 1), 8, 8]
        or [17, 8, 8] in total. The meaning of each plate (channel) input is as follows:
            1. Location of friendly pawns encoded with 1s
            2. Location of friendly knights encoded with 1s
            3. Location of friendly bishops encoded with 1s
            4. Location of friendly rooks encoded with 1s
            5. Location of friendly queens encoded with 1s
            6. Location of friendly king encoded with a 1
            7. - 12. Location of foe pawns, knights, biships, rooks, queens, king
            13. If player's side can king-side castle (all 1s if yes, otherwise 0s)
            14. If player's side can queen-side castle (all 1s if yes, otherwise 0s)
            15. & 16. Same for opponent's king-side and queen-side castling rights
            17. Fifty-move rule counter - Encodes the number of half-moves since last capture or pawn move

        Locationally, the friendly pieces are always shown at the bottom of the board and foe pieces always
        are shown at the top of the board. Therefore, there is some board flipping depending on the color
        whose move it is next so that the model learns the same chess playing from both perspectives.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, 17, 8, 8)
        """
        sym_to_int = {s:i for i, s in enumerate("pnbrqk")}  # Mapping from symbol e.g. "b" to index e.g. 2
        output = torch.zeros((len(state_batch), 17, 8, 8))  # 17 channels, 8 rows, 8 cols

        for i, state in enumerate(state_batch):  # Add each state in the batch to the output torch.Tensor
            board = chess.Board(state)  # Use the FEN string encoding to create the board
            for cell, piece in board.piece_map().items():  # Iter over the piece dictionary
                r, c = divmod(cell, 8)  # Get the row and column of this piece
                if board.turn:  # By default, White's pieces will be in cells 0, 1, ... etc. and black's will
                    # be in cells ... 63, 64 etc. We want to have friendly pieces always in cells (7, ...)
                    # so if it's black's turn to move, we will leave the rows as-is, otherwise if it's white's
                    # turn to move, then we will horizontally flip the rows so that the model always sees
                    # friendly pieces in the bottom rows and foe pieces in the top rows
                    r = 7 - r # Invert the row value so that friendly pieces are always in the bottom rows
                p = sym_to_int[piece.symbol().lower()] # Convert from piece symbol (str) to integer [0, 5]
                if board.turn:  # If it is white's move, then upper-case (white) are the player's pieces
                    # which we will record in the indcies [0, 5] of the third dimension
                    p = p + (6 if piece.symbol().islower() else 0)
                else: # If it is black's move, then upper-case (white) pieces are the opponent's
                    p = p + (0 if piece.symbol().islower() else 6)
                output[i, p, r, c] = 1 # Record using one-hot-encoding

            # Add additional plates for encoding other important state information

            # A). Add castling rights on the king and queen side for friendly and foe
            if board.turn is chess.WHITE: # If it is white's turn, then white is friendly and black is foe
                friendly_color, foe_color = chess.WHITE, chess.BLACK
            else: # Otherwise if it is black's turn, then black is friendly and white is foe
                friendly_color, foe_color = chess.BLACK, chess.WHITE

            output[i, 12, :, :] = 1 if board.has_kingside_castling_rights(friendly_color) else 0
            output[i, 13, :, :] = 1 if board.has_queenside_castling_rights(friendly_color) else 0
            output[i, 14, :, :] = 1 if board.has_kingside_castling_rights(foe_color) else 0
            output[i, 15, :, :] = 1 if board.has_queenside_castling_rights(foe_color) else 0

            # B). 50 move rule counter - Encodes the number of half-moves since last capture or pawn move
            # when this counter reaches 100, then 50 whole moves have been made and the game ends in a draw
            output[i, 16, :, :] = min(board.halfmove_clock, 100) / 100.0 # Scale to be [0, 1]

        # Move the model_input to the required device so it can be run through the network before returning
        return output.to(self.device)

    def forward(self, state_batch: List[str]) -> torch.Tensor:
        """
        Forward pass through the model which generates a value estimate of the current board position for
        each state observation in the input state_batch i.e. an estimate of the expected reward from the
        current state position.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, ) with the value estimates for each stating position.
        """
        # Convert the input board into the expected state representation and pass it through the network
        return self.v_network(self.state_to_model_input(state_batch))


############################################
### Transformer Value-Network Definition ###
############################################
# TODO: Section marker

class Transformer(DVN):
    """
    Implementation of a transformer model for board value estimation.
    """
    def initialize_model(self) -> None:
        """
        ADD MORE HERE

        1). Add positional encodings to the batch of tensors and add a CLS token
        2). Use a few self attention blocks to process the input, essentially an encoder-only model
        3). Use a linear layer on the CLS token (65) to make a final value projection -> tanh()
        """
        pass
        ## TODO: Add here

    def state_to_model_input(self, state_batch: List[str]) -> torch.Tensor:
        """
        ADD MORE HERE
        """
        # Create 6 friendly tokens for each piece type, 6 foe tokens for each piece type, no blank token,
        # 64 positional embedding tokens. So we should output something that is (batch_size, 64) where each
        # element of the 2nd dimension is an integer denoting a different cell on the board. Use integers
        # [0, 11] to represent each 0-5 = friendly, 6-11 = foe
        # then that should allow us to easily pass each into an embedding layer in the transformer to create
        # (batch_size, 64, 128) sized embedding representations
        # Then we should be able to pass that all through say 3 blocks of self attention and then get out
        # a value score at the end by doing an average pooling across the non-empty squares or something like
        # that. Other approaches use the king token pooling only or global averaging across all cells but
        # downweight the empty ones a bit
        pass
        ## TODO: Add here

    def forward(self, state_batch: List[str]) -> torch.Tensor:
        """
        Forward pass through the model which generates a value estimate of the current board position for
        each state observation in the input state_batch i.e. an estimate of the expected reward from the
        current state position.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, ) with the value estimates for each stating position.
        """
        # Convert the input board into the expected state representation and pass it through the network
        return self.v_network(self.state_to_model_input(state_batch))
