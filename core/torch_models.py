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
from utils.general import compute_img_out_dim
from typing import Tuple, List, Dict


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
        self.device = next(self.model.parameters()).device

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
        extra_info = torch.zeros((len(state_batch), 5))  # 4 castling rights + 1 repition counter
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
        # Convert the input board into the expected state representation and pass it through the network
        return self.model(self.state_to_model_input(state_batch))


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
            nn.LeakyReLU(),  # (batch_size, 64, 8, 8)
            ResBlockCNN(64),  # (batch_size, 64, 8, 8)

            # Conv2d Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
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
        self.device = next(self.model.parameters()).device

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
        # Convert the input board into the expected state representation and pass it through the network
        return self.model(self.state_to_model_input(state_batch))


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
        self.device = next(self.encoder.parameters()).device

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
        x = self.state_to_model_input(state_batch)  # (batch_size, ) -> (batch_size, 68, embed_size)
        x = self.encoder(x)  # Pass x through the encoder blocks, (batch_size, 68, embed_size)
        # Perform global average pooling across all the output attention vectors to get a final vector
        x = x.mean(axis=1)  # (batch_size, 68, embed_size) -> (batch_size, embed_size)
        return F.tanh(self.proj(x))  # Linear projection to 1 final output value estimate [-1, +1]


####################################
### Chess Agent Class Definition ###
####################################
# TODO: Section marker

class ChessAgent(DVN):
    """
    Wrapper class around DVN that creates a deep value network model with a PyTorch based self.v_network
    attribute for computing value estimates of input states.
    """

    def initialize_model(self) -> None:
        """
        Initializes the required value network model as a fully-connected neural network. This network
        architecture follows that of the TD-λ evaluator.

        The input to this network will be a batch of FEN string state representation of the current game
        state. self.v_network will be the learned value-function approximator. This method also instantiates
        an optimizer for training.
        """
        if self.config["model_class"] == "MLP":
            model_class = MLP
        elif self.config["model_class"] == "CNN":
            model_class = CNN
        elif self.config["model_class"] == "Transformer":
            model_class = Transformer
        else:
            raise ValueError(f"Model type from config not recognized: {self.config['model']['type']}")

        self.v_network = model_class(self.config)  # Init the value network using the config file

        # Init the optimizer for training, add L2 regularization through the weight_decay parameter
        self.optimizer = torch.optim.Adam(self.v_network.parameters(),
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
