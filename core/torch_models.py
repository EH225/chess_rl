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
import torch.nn.functional as F
from core.base_components import DVN
import core.search_algos as search_algos
from utils.chess_env import ChessEnv, create_ep_record, relative_material_diff
from typing import Tuple, List, Dict

torch.backends.mkldnn.enabled = True # Usually enabled, but set to be sure

##################################################
### Pre-Training Material Heuristic Definition ###
##################################################
# TODO: Section marker

class MaterialHeuristic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = "cpu"  # Alwys run on the CPU
        self.pos_embeddings = nn.Parameter(torch.zeros(1)) # Needs a parameter for the optimizer init

    def forward(self, state_batch: List[str]) -> torch.Tensor:
        """
        Forward pass through the model which generates a value estimate of the current board position for
        each state observation in the input state_batch i.e. an estimate of the expected reward from the
        current state position.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, ) with the value estimates for each stating position.
        """
        if len(state_batch) > 0:
            return torch.Tensor([relative_material_diff(state) for state in state_batch])
        else:  # If an empty batch is passed, return an empty torch.Tensor
            return torch.zeros(0).to(self.device)


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
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.fc1(x))
        h = self.fc2(h)
        return self.activation(x + h)


class MLP(nn.Module):
    """
    Implementation of a multi-layer perceptron (MLP) / fully-connected neural network (FCNN) model for chess
    board value estimation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes a value network model as a fully-connected neural network with residual connections. This
        network architecture follows that of the TD-λ evaluator.

        The input to this network will be a batch of FEN string state representation of the current game
        state and output a torch.Tensor of the same length detailing the model's value estimates for each
        input state.
        """
        super().__init__()
        input_shape = 8 * 8 * 6 + 4 + 1  # 8 rows, 8 cols, 6 piece types, -1, 0, 1 values denoting a piece as
        # friendly or foe or if the cell is empty for a total size of 384 input features +4 castling rights
        # and +1 50-move rule counter (draw condition)
        self.model = nn.Sequential(
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
        )  # Use a final Tanh activation function at the end to produce value estimates [-1, +1]
        self.device = next(self.model.parameters()).device.type

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
        whose move it is next so that the model learns the same chess playing from the same perspective.

        The [8 x 8 x 6] board tensors are flattened and then a few additional info nodes are appended to
        indicate if each side can still king-side castle or queen-side castle and how close the current match
        is to reaching the 50 move draw counter (if no pawn moves or captures).

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, (8 * 8 * 6 + 4 + 1)) = (batch_size, 389)
        """
        sym_to_int = {s: i for i, s in enumerate("pnbrqk")}  # Mapping from symbol e.g. "b" to index e.g. 2
        output = torch.zeros((len(state_batch), 8, 8, 6))  # 8 rows, 8 cols, 6 piece types, encode -1, 0, or 1
        extra_info = torch.zeros((len(state_batch), 5))  # 4 castling rights + 1 repetition counter
        for i, state in enumerate(state_batch):
            board = chess.Board(state)  # Use the FEN string encoding to create the board
            for cell, piece in board.piece_map().items():  # Iter over the piece dictionary
                r, c = divmod(cell, 8)  # Get the row and column of this piece
                # By default, White's pieces will be in cells 0, 1, ... etc. and black's will be in cells
                # ... 63, 64 etc. We want to have friendly pieces always in row 7 so if its black's turn to
                # move, we will flip the columns only so that cell 64 (h8) maps to the bottom left corner. On
                # white's move, we will flip the rows only so that cell 0 (a1) maps to the bottom left corner
                if board.turn:  # White's turn to move
                    r = 7 - r  # Reverse the row order
                else:  # Black's turn to move
                    c = 7 - c  # Reverse the col order
                if board.turn:  # If it is white's move, then upper-case (white) are the player's pieces
                    val = -1 if piece.symbol().islower() else 1
                else:  # If it is black's move, then upper-case (white) pieces are the opponent's
                    val = 1 if piece.symbol().islower() else -1
                output[i, r, c, sym_to_int[piece.symbol().lower()]] = val

            # A). Add castling rights on the king and queen side for friendly and foe
            if board.turn is chess.WHITE:  # If it is white's turn, then white is friendly and black is foe
                friendly_color, foe_color = chess.WHITE, chess.BLACK
            else:  # Otherwise if it is black's turn, then black is friendly and white is foe
                friendly_color, foe_color = chess.BLACK, chess.WHITE
            extra_info[i, 0] = 1 if board.has_kingside_castling_rights(friendly_color) else 0
            extra_info[i, 1] = 1 if board.has_queenside_castling_rights(friendly_color) else 0
            extra_info[i, 2] = 1 if board.has_kingside_castling_rights(foe_color) else 0
            extra_info[i, 3] = 1 if board.has_queenside_castling_rights(foe_color) else 0

            # B). 50 move rule counter - Encodes the number of half-moves since last capture or pawn move
            # when this counter reaches 100, then 50 whole moves have been made and the game ends in a draw
            extra_info[i, 4] = min(board.halfmove_clock, 100) / 100.0  # Scale to be [0, 1]

        output = output.flatten(start_dim=1)  # (batch_size, 8, 8, 6) -> (batch_size, 384)
        # Add info about castling rights and how close are to a draw based on the 50-move rule
        output = torch.concatenate([output, extra_info], dim=1)  # (B, 384) +  (B, 5) =  (B, 389)
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
        if len(state_batch) > 0:
            # Convert the input board into the expected state representation and pass it through the network
            return self.model(self.state_to_model_input(state_batch)).squeeze(1)
        else:  # If an empty batch is passed, return an empty torch.Tensor
            return torch.zeros(0).to(self.device)


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
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.activation(x + h)


class CNN(nn.Module):
    """
    Implementation of a convolutional neural network model for chess board value estimation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the required value network model as a convolutional neural network (CNN). This network
        architecture follows a blend of various other resnet-based CNNs including the one from AlphaZero.

        The input to this network will be a batch of FEN string state representation of the current game
        state and output a torch.Tensor of the same length detailing the model's value estimates for each
        input state.
        """
        super().__init__()
        self.model = nn.Sequential(
            # Conv2d Block 1
            nn.Conv2d(in_channels=17, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # (batch_size, 64, 8, 8)
            ResBlockCNN(64),  # (batch_size, 64, 8, 8)

            # Conv2d Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),  # (batch_size, 128, 8, 8)
            ResBlockCNN(128),  # (batch_size, 128, 8, 8)

            # Dense fully-connected Block 3
            nn.Flatten(),  # (batch_size, 128, 8, 8) -> (batch_size, 8192)
            nn.Linear(128 * 8 * 8, 512),  # (batch_size, 8192) -> (batch_size, 512)
            nn.LeakyReLU(),
            nn.Linear(512, 128),  # (batch_size, 512) -> (batch_size, 128)
            nn.LeakyReLU(),
            nn.Linear(128, 1),  # (batch_size, 128) -> (batch_size, 1)
            nn.Tanh()
        )  # Use a final Tanh activation function at the end to produce value estimates [-1, +1]
        self.device = next(self.model.parameters()).device.type

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
        whose move it is next so that the model learns the same chess playing from the same perspective.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, 17, 8, 8)
        """
        sym_to_int = {s: i for i, s in enumerate("pnbrqk")}  # Mapping from symbol e.g. "b" to index e.g. 2
        output = torch.zeros((len(state_batch), 17, 8, 8))  # 17 channels, 8 rows, 8 cols

        for i, state in enumerate(state_batch):  # Add each state in the batch to the output torch.Tensor
            board = chess.Board(state)  # Use the FEN string encoding to create the board
            for cell, piece in board.piece_map().items():  # Iter over the piece dictionary
                r, c = divmod(cell, 8)  # Get the row and column of this piece
                # By default, White's pieces will be in cells 0, 1, ... etc. and black's will be in cells
                # ... 63, 64 etc. We want to have friendly pieces always in row 7 so if its black's turn to
                # move, we will flip the columns only so that cell 64 (h8) maps to the bottom left corner. On
                # white's move, we will flip the rows only so that cell 0 (a1) maps to the bottom left corner
                if board.turn:  # White's turn to move
                    r = 7 - r  # Reverse the row order
                else:  # Black's turn to move
                    c = 7 - c  # Reverse the col order
                p = sym_to_int[piece.symbol().lower()]  # Convert from piece symbol (str) to integer [0, 5]
                if board.turn:  # If it is white's move, then upper-case (white) are the player's pieces
                    # which we will record in the indcies [0, 5] of the third dimension
                    p = p + (6 if piece.symbol().islower() else 0)
                else:  # If it is black's move, then upper-case (white) pieces are the opponent's
                    p = p + (0 if piece.symbol().islower() else 6)
                output[i, p, r, c] = 1  # Record using one-hot-encoding

            # Add additional plates for encoding other important state information

            # A). Add castling rights on the king and queen side for friendly and foe
            if board.turn is chess.WHITE:  # If it is white's turn, then white is friendly and black is foe
                friendly_color, foe_color = chess.WHITE, chess.BLACK
            else:  # Otherwise if it is black's turn, then black is friendly and white is foe
                friendly_color, foe_color = chess.BLACK, chess.WHITE

            output[i, 12, :, :] = 1 if board.has_kingside_castling_rights(friendly_color) else 0
            output[i, 13, :, :] = 1 if board.has_queenside_castling_rights(friendly_color) else 0
            output[i, 14, :, :] = 1 if board.has_kingside_castling_rights(foe_color) else 0
            output[i, 15, :, :] = 1 if board.has_queenside_castling_rights(foe_color) else 0

            # B). 50 move rule counter - Encodes the number of half-moves since last capture or pawn move
            # when this counter reaches 100, then 50 whole moves have been made and the game ends in a draw
            output[i, 16, :, :] = min(board.halfmove_clock, 100) / 100.0  # Scale to be [0, 1]

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
        if len(state_batch) > 0:
            # Convert the input board into the expected state representation and pass it through the network
            return self.model(self.state_to_model_input(state_batch)).squeeze(1)
        else:  # If an empty batch is passed, return an empty torch.Tensor
            return torch.zeros(0).to(self.device)


############################################
### Transformer Value-Network Definition ###
############################################
# TODO: Section marker

class Transformer(nn.Module):
    """
    Implementation of a multi-headed self-attention transformer model for chess board value estimation.
    """

    def __init__(self, config, *args, **kwargs):
        """
        Initializes a value network model as a transformer that follows the architecture of the original
        transformer paper "Attention is All You Need" (https://arxiv.org/abs/1706.03762) with multiple
        layers of self-attention blocks followed by an average pooling and linear projection layer.

        The input to this network will be a batch of FEN string state representation of the current game
        state and output a torch.Tensor of the same length detailing the model's value estimates for each
        input state.
        """
        super().__init__()
        # Extract config parameters from the passed model config dictionary
        self.embed_size = int(config["model"]["embed_size"])
        self.hidden_size = int(config["model"]["hidden_size"])
        self.n_heads = int(config["model"]["n_heads"])
        self.num_layers = int(config["model"]["num_layers"])
        self.ff_dim = int(config["model"]["ff_dim"])

        # Create a token-embedding layer for the pieces and castling rights. We have 1 token for blank
        # squares, 6 for friendly pieces, 6 for foe pieces, and 2 castling rights for each player, each with
        # 2 levels (True and False) for a total of 1 + 6 + 6 + 4 * 2 = 21 unique token integer indices
        self.token_embeddings = nn.Embedding(num_embeddings=21, embedding_dim=self.embed_size, padding_idx=0)
        # Create a matrix of size [64, embed_size] of learnable parameters which will serve as the positional
        # embeddings for each unique square on the board and will be added to the piece encodings
        self.pos_embeddings = nn.Parameter(torch.zeros(1, 64, self.embed_size))
        # Create the multi-headed self-attention transformer blocks, the core of the transformer model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_heads,
                                                        dim_feedforward=self.ff_dim, activation="relu",
                                                        batch_first=True, norm_first=False, bias=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers,
                                             norm=nn.LayerNorm(self.hidden_size))
        self.proj = nn.Linear(self.embed_size, 1)  # Final linear projection after pooling to 1 output value
        self.device = next(self.encoder.parameters()).device.type

    def state_to_model_input(self, state_batch: List[str]) -> torch.Tensor:
        """
        Converts an input batch of board states (encoded using FEN as a string) into the expected state
        representations for this model (torch.Tensor) and moves the data to the same device as this model.

        This transformer model operates on board representations that are (64 + 4, embed_size) = (68, E) in
        size. The first 64 entries of the first dimension are the 64 positional embedding vectors of the board
        plus their respective piece token embeddings (if non-empty, empty squares use only a positional
        embedding vector). The last 4 entries along the first dimension are token embeddings for the 4
        castling rights variables i.e. white king-side, white queen-side, black king-side, black queen-side.
        For each castling rights token, there are 2 possible options, one representing True and the other
        representing False. Each token embedding is a length E (e.g. 128) vector and there will always be a
        fixed number (i.e. 68) passed in per batch element (board state).

        Locationally, the friendly pieces are always shown at the bottom of the board and foe pieces always
        are shown at the top of the board. Therefore, there is some board flipping depending on the color
        whose move it is next so that the model learns the same chess playing from the same perspective.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, 68, E) containing vector embeddings.
        """
        sym_to_int = {s: (i + 1) for i, s in enumerate("pnbrqk")}  # Mapping from symbol to int starting at 1
        output = torch.zeros((len(state_batch), 8, 8), dtype=torch.int)  # 8 x 8 = 64 squares on a chess
        # board, use 0 to denote blank squares, [1, 6] for the friendly pieces and [7, 12] for foe
        castling = torch.zeros(len(state_batch), 4, dtype=torch.int)  # 4 possible castling rights

        for i, state in enumerate(state_batch):  # Add each state in the batch to the output torch.Tensor
            board = chess.Board(state)  # Use the FEN string encoding to create the board
            for cell, piece in board.piece_map().items():  # Iter over the piece dictionary
                r, c = divmod(cell, 8)  # Get the row and column of this piece
                # By default, White's pieces will be in cells 0, 1, ... etc. and black's will be in cells
                # ... 63, 64 etc. We want to have friendly pieces always in row 7 so if its black's turn to
                # move, we will flip the columns only so that cell 64 (h8) maps to the bottom left corner. On
                # white's move, we will flip the rows only so that cell 0 (a1) maps to the bottom left corner
                if board.turn:  # White's turn to move
                    r = 7 - r  # Reverse the row order
                else:  # Black's turn to move
                    c = 7 - c  # Reverse the col order
                p = sym_to_int[piece.symbol().lower()]  # Convert from piece symbol (str) to integer [1, 6]
                if board.turn:  # If it is white's move, then upper-case (white) are the player's pieces
                    # which we will record with piece tokens [1, 6], otherwise record them with [7, 12] ints
                    p = p + (6 if piece.symbol().islower() else 0)
                else:  # If it is black's move, then upper-case (white) pieces are the opponent's pieces,
                    # which we will record with piece tokens [1, 6], otherwise record them with [7, 12] ints
                    p = p + (6 if piece.symbol().isupper() else 0)
                output[i, r, c] = p  # Record the piece token int [1, 12], leave zero for blank cells

            # Encode the castling rights
            if board.turn is chess.WHITE:  # If it is white's turn, then white is friendly and black is foe
                friendly_color, foe_color = chess.WHITE, chess.BLACK
            else:  # Otherwise if it is black's turn, then black is friendly and white is foe
                friendly_color, foe_color = chess.BLACK, chess.WHITE

            # Encode the True and False of each castling right for each player as a separate token value
            castling[i, 0] = 13 if board.has_kingside_castling_rights(friendly_color) else 14
            castling[i, 1] = 15 if board.has_queenside_castling_rights(friendly_color) else 16
            castling[i, 2] = 17 if board.has_kingside_castling_rights(foe_color) else 18
            castling[i, 3] = 19 if board.has_queenside_castling_rights(foe_color) else 20

        # Output is now (batch_size, 8, 8) and has the integers [1, 6] for friendly pieces, [7, 12] for foe
        # pieces and 0s for the empty squares which will be treated as padding tokens by the embedding layer
        output = output.reshape(len(state_batch), -1)  # Reshape to (batch_size, 64) to flatten

        # Add the additional 4 tokens for castling rights and move to the required device
        output = torch.concat([output, castling], dim=1).to(self.device)  # (batch_size, 68) ints

        # Pass these token integers through the embedding layer to convert them into embedding vectors
        output = self.token_embeddings(output)  # (batch_size, 68, embed_size)

        # Add positional encodings to the first 64 elements corresponding to tiles on the chess board
        output[:, :64, :] += self.pos_embeddings

        return output  # (batch_size, 68, embed_size)

    def forward(self, state_batch: List[str]) -> torch.Tensor:
        """
        Forward pass through the model which generates a value estimate of the current board position for
        each state observation in the input state_batch i.e. an estimate of the expected reward from the
        current state position.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, ) with the value estimates for each stating position.
        """
        if len(state_batch) > 0:
            x = self.state_to_model_input(state_batch)  # (batch_size, ) -> (batch_size, 68, embed_size)
            x = self.encoder(x)  # Pass x through the encoder blocks, (batch_size, 68, embed_size)
            # Perform global average pooling across all the output attention vectors to get a final vector
            x = x.mean(axis=1)  # (batch_size, 68, embed_size) -> (batch_size, embed_size)
            # Linear projection to 1 final output value estimate [-1, +1]
            return F.tanh(self.proj(x)).squeeze(1)
        else:  # If an empty batch is passed, return an empty torch.Tensor
            return torch.zeros(0).to(self.device)


####################################
### Chess Agent Class Definition ###
####################################
# TODO: Section marker

class ChessAgent(DVN):
    """
    Wrapper class around DVN that creates a deep value network model with a PyTorch based self.v_network
    attribute for computing value estimates of input states.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = self.config["model"]
        self.model_class = self.config["model_class"]

    def initialize_model(self) -> None:
        """
        Initializes the required value network model as a fully-connected neural network. This network
        architecture follows that of the TD-λ evaluator.

        The input to this network will be a batch of FEN string state representation of the current game
        state. self.v_network will be the learned value-function approximator. This method also instantiates
        an optimizer for training.
        """
        try:
            model_class = globals()[self.config["model_class"]]
        except:
            raise ValueError(f"Model type from config not recognized: {self.config['model_class']}")

        self.v_network = model_class(self.config)  # Init the value network using the config file

        # Init the optimizer for training, add L2 regularization through the weight_decay parameter
        self.optimizer = torch.optim.AdamW(self.v_network.parameters(),
                                           lr=self.config["hyper_params"]["lr_begin"],
                                           weight_decay=self.config["hyper_params"]["wt_decay"])

    def forward(self, state_batch: List[str]) -> torch.Tensor:
        """
        Forward pass through the value network model which generates a value estimate of the current board
        position from the perspective of the player to move next for each state observation in the input
        state_batch i.e. an estimate of the expected reward from the current state position.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, ) with the value estimates for each stating position.
        """
        return self.v_network(state_batch)

    def __call__(self, state_batch: List[str]) -> torch.Tensor:
        return self.forward(state_batch)

    @staticmethod
    def _compute_td_targets(state_batch: List[str], config: Dict, t: int) -> Dict[str, np.ndarray]:
        """
        This method sequentially computes the estimated value of each state in state_batch using the search
        function and model specified in the input config dictionary. A model instance is instantiated and the
        weights are read in from disk. Each state estimate is then computed using a search function and the
        results are collected and returned in a np.ndarray. This function is designed to be called in parallel
        using dask.

        :param state_batch: A batch of FEN states as a list of strings.
        :param config: A config dictionary read from yaml that specifies hyperparameters.
        :param t: The current training time step. This is used to control which search function and value
            estimator is used when generating the TD target values.
        :return: A dictionary of np.ndarrays of the same size as state_batch with state value estimates for
            each state, the number of nodes evaluated, the max depth of each search tree, and the number of
            tree nodes evaluated.
        """
        if isinstance(state_batch, str):  # Accept a lone string, convert it to a list of size 1
            state_batch = [state_batch, ]  # All lines below expect state_batch to be a list

        # 1). Init the right kind of v_network model according to the config passed
        if t < config["pre_train"]["nsteps_pretrain"]:
            v_network = MaterialHeuristic()  # Use the material heuristic value estimator during pre-training
        else:
            v_network = globals()[config["model_class"]](config)

        # 2). Load in the weights of the model
        if t >= config["pre_train"]["nsteps_pretrain"]:  # No weight loading needed during pre-training,
            # the value estimator model used is the heuristic model which has no trainable parameters
            if config["model_training"]["local_cluster_only"] is False:
                # Load in the model weights from the local dask working directory, which were broadcasted out
                # since local_cluster_only is False, the weights are uploaded and distributed via dask
                worker_temp_dir = os.path.join(PARENT_DIR, "dask-scratch-space")
                subfolders = set([x for x in os.listdir(worker_temp_dir)
                                  if os.path.isdir(os.path.join(worker_temp_dir, x))])
                if config["model_training"]["use_scripted_model"]:
                    wts_path = os.path.join(config["model_training"]["load_dir"], "model_scripted.bin")
                    v_network.model = torch.jit.load(wts_path, map_location="cpu")
                else:  # If not loading a pre-compiled model, then load in the state dictionary
                    wts_path = os.path.join(worker_temp_dir, subfolders.pop(), "model.bin")
                    v_network.load_state_dict(torch.load(wts_path, map_location="cpu", weights_only=True))
            else:  # Otherwise, read the model weights in from the local weights save directory location
                # i.e. 1 location, not the dask-worker space, no dask_client.upload_file used
                wts_path = os.path.join(config["model_training"]["load_dir"], "model.bin")
                v_network.load_state_dict(torch.load(wts_path, map_location="cpu", weights_only=True))

        # 3). Determine the search function specified by the config
        if t < config["pre_train"]["nsteps_pretrain"]:
            search_func = getattr(search_algos, config["pre_train"]["name"])
            search_func_kwargs = config["pre_train"]
        else:
            search_func = getattr(search_algos, config["search_func"]["name"])
            search_func_kwargs = config["search_func"]

        # 4). Compute the TD targets sequentially using the search function
        state_values = np.zeros(len(state_batch), dtype=np.float32)  # 1 float state estimate per state
        total_nodes = np.zeros(len(state_batch), dtype=np.int32)
        max_depths = np.zeros(len(state_batch), dtype=np.int32)
        terminal_nodes = np.zeros(len(state_batch), dtype=np.int32)
        for i, state in enumerate(state_batch):
            _, state_value, _, info = search_func(state=state, model=v_network, **search_func_kwargs)
            state_values[i] = state_value
            total_nodes[i] = info[0]
            max_depths[i] = info[1]
            terminal_nodes[i] = info[2]

        return {"state_values": state_values, "total_nodes": total_nodes,
                "max_depths": max_depths, "terminal_nodes": terminal_nodes}

    @staticmethod
    def _run_game(epsilon: float, config: Dict, *args, **kwargs) -> Tuple[List[str], Dict]:
        """
        Runs 1 on-policy self-play chess match with an epsilon greedy action selection strategy. This function
        is designed to be called in parallel using dask to simulate many games simultaneously to generate
        state observations for the replay buffer during training.

        :param epsilon: The exploration parameter i.e. with probability e the agent selects a random action.
        :param config: A config dictionary read from yaml that specifies hyperparameters.
        :return: A list of game states (a list of FEN strings) and an ep_record summarizing the game.
        """
        # 1). Init the right kind of v_network model according to the config passed
        v_network = globals()[config["model_class"]](config)

        # 2). Load in the weights of the model for on-policy state generation
        if config["model_training"]["local_cluster_only"] is False:
            # Load in the model weights from the local dask working directory, which were broadcasted out
            # since local_cluster_only is False, the weights are uploaded and distributed via dask
            worker_temp_dir = os.path.join(PARENT_DIR, "dask-scratch-space")
            subfolders = set([x for x in os.listdir(worker_temp_dir)
                              if os.path.isdir(os.path.join(worker_temp_dir, x))])
            if config["model_training"]["use_scripted_model"]:
                wts_path = os.path.join(config["model_training"]["load_dir"], "model_scripted.bin")
                v_network.model = torch.jit.load(wts_path, map_location="cpu")
            else:  # If not loading a pre-compiled model, then load in the state dictionary
                wts_path = os.path.join(worker_temp_dir, subfolders.pop(), "model.bin")
                v_network.load_state_dict(torch.load(wts_path, map_location="cpu", weights_only=True))
        else:  # Otherwise, read the model weights in from the local weights save directory location
            # i.e. 1 location, not the dask-worker space, no dask_client.upload_file used
            wts_path = os.path.join(config["model_training"]["load_dir"], "model.bin")
            v_network.load_state_dict(torch.load(wts_path, map_location="cpu", weights_only=True))

        # 3). Create a chess env and prepare recording variables
        env = ChessEnv()
        state = env.reset()
        states = [state]  # Record all the states reached during game play

        # 4). Determine the search function specified by the config
        search_func = getattr(search_algos, config["search_func"]["name"])

        # 5). Run a self-play game until truncation or termination
        while True:  # Run until the episode has finished
            if np.random.rand() < epsilon:  # With probability epsilon, choose a random action
                action, state_value, action_values = env.action_space.sample(), 0, np.zeros(0)
            else:  # Otherwise actually use the model to evaluate
                action, state_value, action_values, info = search_func(state=state, model=v_network,
                                                                       **config["search_func"])

            # Perform the selected action in the env, get the new state, reward, and stopping flags
            new_state, reward, terminated, truncated = env.step(action)

            states.append(new_state)
            state = new_state  # Update for next iteration, move to the new FEN state representation
            # End the episode if one of the stopping conditions is met
            if terminated or truncated:
                break

        # 5). Create a new episode record from the move stack of the game
        ep_record = create_ep_record(env.board.move_stack)

        return states, ep_record
