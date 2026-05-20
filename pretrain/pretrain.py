"""
This module contains the source code for running supervised pre-training of a policy-value network.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import numpy as np
import torch
import argparse
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim import AdamW
from typing import Tuple, Callable, Dict, List
import logging
import chess
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from utils.general import get_device, get_amp_dtype
from core.torch_models import MLP, CNN, Transformer
from utils.general import read_yaml, create_move_to_idx_map




class SupervisedPretrainingDataset(Dataset):
    """
    A dataset for supervised pre-training of a policy-value network on stockfish eval scores and selected
    moves from the lichess database of 38mm games.
    """

    def __init__(self, dataset_path: str, state_to_model_input: Callable):
        """
        Initializes a dataset object for supervised pre-training (SPT) of a policy-value network.

        :param dataset_path: A file path for where the supervised pre-training parquet dataset is cached.
        :param state_to_model_input: A callable function that converts a batch of FEN board state encodings
            into a torch.Tensor that can be passed into v_network.
        """
        self.dataset = pd.read_parquet(dataset_path)  # Read in the data as a pd.DataFrame, 2.6 GB
        self.state_to_model_input = state_to_model_input  # Function for converting a FEN into a tensor

    def __len__(self):
        """
        Returns the total number of observations in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary containing keys "boards" and "eval_tgts" for a particular index in the dataset.
        """
        return {
            "fen_states": self.dataset.loc[idx, "fen"],
            "state_tensors": self.state_to_model_input([self.dataset.loc[idx, "fen"]]).squeeze(0),
            "value_tgt": torch.tensor(self.dataset.loc[idx, "value_tgt"], dtype=torch.float32),
            "policy_tgt": torch.tensor(self.dataset.loc[idx, "policy_tgt"], dtype=torch.long),
        }


def get_dataloader(batch_size: int, dataset_path: str, state_to_model_input: Callable) -> DataLoader:
    """
    Creates a dataloader for supervised pre-training of a value network on stockfish eval scores.

    :param batch_size: The size of the batches to return from the dataloader.
    :param dataset_path: A file path for where the supervised pre-training parquet dataset is cached.
    :param state_to_model_input: A callable function that converts a batch of FEN board state encodings
        into a torch.Tensor that can be passed into v_network.
    :returns: A pytorch DataLoader object.
    """
    device = get_device()  # Auto-detect the available hardware
    dataset = SupervisedPretrainingDataset(dataset_path, state_to_model_input)
    if device == "cuda":
        num_workers, pin_memory, persistent_workers = os.cpu_count(), True, True
    else:
        num_workers, pin_memory, persistent_workers = 0, False, False
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=4)


def infinite_loader(dataloader: DataLoader):
    """
    Infinitely yields batches of data from the input dataloader (dl) without caching batches.
    """
    while True:
        for batch in dataloader:
            yield batch


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


class Trainer:
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
                 lr_start: float = 1e-4, lr_end: float = 1e-5, weight_decay: float = 1e-3,
                 train_num_steps: int = 300000, adam_betas: Tuple[float] = (0.9, 0.99),
                 grad_clip: float = 1.0, eval_every: int = 5000, save_every: int = 10000,
                 results_folder: str = None, use_amp: bool = False, use_latest_checkpoint: bool = True,
                 reset_lr_scheduler: bool = False, **kwargs):
        """
        This is a framework for training a deep policy-value network model. This class wrapper has methods
        for loading a model from a recent checkpoint, saving a model periodically during training, and
        running a training loop to train from scratch or to continue from the last checkpoint.

        :param model: A value network implemented in pytorch.
        :param train_dataloader: A dataloader that will yield the training batches.
        :param val_dataloader: A dataloader that will yield the validation batches.
        :param lr_start: The initial learning rate.
        :param lr_end: The terminal training learning rate.
        :param weight_decay: The weight_decay to provide to the Adam optimizer for L2 regularization.
        :param train_num_steps: The number of training steps to run in total.
        :param adam_betas: Beta parameters for the adam optimizer.
        :param grad_clip: The amount of gradient clipping to use during training.
        :param eval_every: Specifies how often to run evaluation metrics on the validation dataset.
        :param save_every: An int denoting how often to save the model weights and losses.
        :param results_folder: A location to save the results of training.
        :param use_amp: Whether to use automatic mixed-precision type casting during training.
        :param use_latest_checkpoint: If set to True, then the latest checkpoint detected in the results
            directory will be loaded in before training begins to pick up from where it was last left off.
        :param reset_lr_scheduler: If set to True, then the learning rate scheduler is reset if the load
            method is called. This allows us to continue training further by configuring a new learning rate
            annealing scheduler. Note, this makes it so that we do not have a learning rate warm-up. For
            continuity with prior training, it is recommended that the new lr_start == prior lr_end.
        """
        super().__init__()

        assert results_folder is not None, "You must specify results folder to save the outputs"

        self.results_folder = results_folder  # A directory where the checkpoints will be saved
        self.checkpoints_folder = os.path.join(self.results_folder, "checkpoints/")
        self.losses_folder = os.path.join(self.results_folder, "losses/")
        for directory in [self.results_folder, self.checkpoints_folder, self.losses_folder]:
            os.makedirs(directory, exist_ok=True)  # Create the directory if not already there

        # Set up logging during training
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:  # Prevent duplicate handlers
            file_handler = logging.FileHandler(os.path.join(self.results_folder, "train.log"),
                                               encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            # file_handler.stream = sys.stdout  # Ensure UTF-8 capable stream
            self.logger.addHandler(file_handler)

            tqdm_handler = TqdmLoggingHandler()
            tqdm_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            # tqdm_handler.stream = sys.stdout  # Ensure UTF-8 capable stream
            self.logger.addHandler(tqdm_handler)
        self.logger.propagate = False

        self.model = model  # The policy-value network being trained
        self.logger.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
        self.device = get_device()  # Auto-detect what device to use for training
        self.grad_clip = grad_clip  # The amount of gradient clipping to use during training
        self.amp_dtype = get_amp_dtype(self.device) if use_amp else None
        self.eval_every = eval_every # The frequency of validation set evals
        self.save_every = save_every  # The frequency of saving model weights
        self.train_num_steps = train_num_steps  # The total number of training steps

        # Save a pointer to the train and validation dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Configure the optimizer for training
        decay_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in ['bias', 'bn'])]
        no_decay_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and any(nd in n for nd in ['bias', 'bn'])]

        self.opt = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
            ], lr=lr_start, betas=adam_betas)

        self.reset_lr_scheduler = reset_lr_scheduler
        self.lr_start, self.lr_end = lr_start, lr_end
        warmup_steps = 5000  # Slowly ramp up the learning rate from very low to peak
        warmup = LinearLR(self.opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        # Linearly decay the learning rate during training
        decay = LinearLR(self.opt, start_factor=1.0, end_factor=lr_end / lr_start,
                         total_iters=train_num_steps - warmup_steps)
        # Stack both the learning rate warm up and the gradual linear decay into 1 scheduler
        self.scheduler = SequentialLR(self.opt, schedulers=[warmup, decay], milestones=[warmup_steps])

        self.step = 0  # Training step counter
        self.train_losses, self.val_losses = [], []  # Aggregate loss values during training
        self.move_uci_to_idx = create_move_to_idx_map()  # Create a mapping from uci move to integer

        if use_latest_checkpoint:
            checkpoints = os.listdir(self.checkpoints_folder)
            if len(checkpoints) > 0:
                last_checkpoint = max([int(x.replace("model-", "").replace(".pt", "")) for x in checkpoints])
                self.load(last_checkpoint)  # Load in the most recent milestone to continue training


    def save(self, milestone: int) -> None:
        """
        Saves the weights of the model for the current milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None. Writes the weights and losses to disk.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        self.logger.info(f"Saving model to {checkpoint_path}.")
        data = {"step": self.step,
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                }
        torch.save(data, checkpoint_path)
        # Save down all the loss values produced by model training since the last caching
        cols = ["step", "policy_loss", "value_loss", "total_loss"]
        # Convert the train losses to a pd.DataFrame and save down the results
        df = pd.DataFrame(self.train_losses, columns=cols)
        df.to_csv(os.path.join(self.losses_folder, f"train-losses-{milestone}.csv"))
        # Convert the validation losses to a pd.DataFrame and save down the results
        df = pd.DataFrame(self.val_losses, columns=cols)
        df.to_csv(os.path.join(self.losses_folder, f"val-losses-{milestone}.csv"))

    def load(self, milestone: int) -> None:
        """
        Loads in the cached weights from disk for a particular milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None. Weights are loaded into the model.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        self.logger.info(f"Loading model from {checkpoint_path}.")
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)

        # Re-instate the training step counter, model weights, and optimizer state from the checkpoint data
        # read in from disk
        self.step = checkpoint_data["step"]
        self.model.load_state_dict(checkpoint_data["model"])
        self.opt.load_state_dict(checkpoint_data["opt"])
        if self.reset_lr_scheduler:  # If True, do not load the prior learning rate scheduler state
            for g in self.opt.param_groups:  # Make sure the optimizer learning rates match the new scheduler
                g["lr"] = self.lr_start

            # Instead of loading the prior learning rate scheduler from disk, create a new one according to
            # the new config provided to continue traing after the prior end
            self.scheduler = LinearLR(self.opt, start_factor=1.0, end_factor=self.lr_end / self.lr_start,
                                      total_iters=self.train_num_steps - self.step)

        else: # If not resetting the LR scheduler, then load in the state dict to re-store it
            self.scheduler.load_state_dict(checkpoint_data["scheduler"])

        # Losses are not loaded in, they are saved to disk periodically with the model weights and are not
        # needed to continue training. The losses obtained by training will be cached again at the next save

        # Move the model and the optimizer to the same device to continue training or for inference
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)


    def _get_legal_move_mask(self, state_batch: List[str], move_uci_to_idx: Dict) -> torch.Tensor:
        """
        Returns a boolean mask of shape (num_batch, 1968) where True = legal move.

        :param state_batch: A list of fen board states.
        :param move_uci_to_idx: A dictionary mapping UCI moves e.g. "a1a2" to integers e.g. 5.
        :returns: A (batch_size, 1968) mask matching the shape of a policy_logits output denoting which moves
            are legal.
        """
        mask = torch.zeros(len(state_batch), 1968, dtype=torch.bool)
        for i, state in enumerate(state_batch):
            board = chess.Board(state)
            for move in board.legal_moves:
                mask[i, move_uci_to_idx[move.uci()]] = True
        return mask

    def train(self, lambda_val: float = 15.0, mask_illegal_moves: bool = True) -> None:
        """
        Runs the training of the model until completion for self.train_num_steps total training iterations.

        :param lambda_val: A weight parameter applied to the MSE value loss output to scale it to be on a
            similar unit scaling as that of the policy loss. The default value is 15.0 since the MSE loss
            is on average around 0.5 and the cross entropy loss is on average log(1968) = 7.6 so 15x applied
            to the value loss would scale it to also be around 7.6.
        :param mask_illegal_moves: If set to True, then illegal moves are masked with -inf in the policy
            logit model outputs so that probability mass given to illegal moves is not penalized.
        """
        msg = f"Starting Training, device={self.device}, amp_dtype={self.amp_dtype}, lambda_val={lambda_val}"
        self.logger.info(msg)
        for i, param_group in enumerate(self.opt.param_groups):  # Report the learning rate and weight decay
            self.logger.info(f"lr={param_group['lr']}, wd={param_group['weight_decay']}")
            break  # Show for only the first parameter group, assume all are the same

        value_loss_fn = nn.MSELoss()  # Use MSE loss for the value head avg value =~ 0.5
        policy_loss_fn = nn.CrossEntropyLoss()  # Use cross-entropy loss for the policy head, avg val =~ 7.6
        # which is derived from log(1968) =~ 7.6

        self.model.to(self.device)  # Move the model to the correct device
        self.model.train()  # Make sure to set the model to train mode for training

        inf_dataloader = infinite_loader(self.train_dataloader)  # This does not cache batches
        grad_norm = 0.0  # Set a default in-case no grad-norm is used for the progress bar

        if self.amp_dtype == torch.float16 and self.device == 'cuda':
            scaler = torch.amp.GradScaler('cuda')

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:  # Run until all training iterations are complete
                # Get the next training batch and move it to the same device as the model
                batch = next(inf_dataloader)
                state_tensors = batch["state_tensors"].to(self.device, non_blocking=True)
                print("state_tensors.shape", state_tensors.shape)
                print("state_tensors", state_tensors)
                value_tgt = batch["value_tgt"].to(self.device, non_blocking=True)
                policy_tgt = batch["policy_tgt"].to(self.device, non_blocking=True)
                print("policy_tgt.shape", policy_tgt.shape)
                print("policy_tgt", policy_tgt)

                self.opt.zero_grad(set_to_none=True)  # Zero the grads of the opt before computing the loss
                # Compute the forward-pass through the model and compute a tensor that is the same shape
                # along the first 2 dims as captions but also gives the prob dist across the vocab
                if self.amp_dtype is not None:
                    with torch.autocast(device_type=self.device, dtype=self.amp_dtype):
                        policy_logits, value_est = self.model(state_tensors)
                        if mask_illegal_moves:  # If True, mask out illegal moves from the policy logits with
                            # -np.inf so that the model does not get penalized for giving them prob mass
                            mask = self._get_legal_move_mask(batch["fen_states"],
                                                             self.move_uci_to_idx).to(self.device)
                            policy_logits = policy_logits.masked_fill(~mask, float('-inf'))
                        print("policy_logits.shape", policy_logits.shape)
                        print("policy_logits", policy_logits)
                        policy_loss = policy_loss_fn(policy_logits, policy_tgt)
                        print("policy_loss", policy_loss)
                        value_loss = value_loss_fn(value_est, value_tgt)
                        total_loss = policy_loss + value_loss * lambda_val
                else:
                    policy_logits, value_est = self.model(state_tensors)
                    if mask_illegal_moves:  # If True, mask out illegal moves from the policy logits with
                        # -np.inf so that the model does not get penalized for giving them prob mass
                        mask = self._get_legal_move_mask(batch["fen_states"],
                                                         self.move_uci_to_idx).to(self.device)
                        policy_logits = policy_logits.masked_fill(~mask, float('-inf'))
                    policy_loss = policy_loss_fn(policy_logits, policy_tgt)
                    value_loss = value_loss_fn(value_est, value_tgt)
                    total_loss = policy_loss + value_loss * lambda_val

                if self.amp_dtype == torch.float16:
                    scaler.scale(total_loss).backward()
                    if self.grad_clip is not None:
                        scaler.unscale_(self.opt)  # Unscale before clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    scaler.step(self.opt)  # Update the model parameters by taking a gradient step
                    scaler.update()
                else:
                    total_loss.backward()
                    if self.grad_clip is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                                   self.grad_clip)
                    self.opt.step()  # Update the model parameters by taking a gradient step

                pbar.set_postfix(
                    policy_loss=f"{policy_loss.item():.4f}", value_loss=f"{value_loss.item():.4f}",
                    total_loss=f"{total_loss.item():.4f}", grad=f"{grad_norm:.3f}")

                self.scheduler.step()  # Update the learning rate scheduler

                # Aggregate all the loss values for each timestep, record separately for each
                self.train_losses.append((self.step, policy_loss.item(),
                                          value_loss.item(), total_loss.item()))
                self.step += 1

                # Periodically run evaluation metrics on the validation data set, always on the last iter too
                if self.step % self.eval_every == 0 or self.step == self.train_num_steps:
                    self.model.eval() # Set model to evaluation mode
                    with torch.no_grad():
                        n_obs_total, policy_loss, value_loss = 0.0, 0.0, 0.0
                        for batch in self.val_dataloader:
                            state_tensors = batch["state_tensors"].to(self.device, non_blocking=True)
                            value_tgt = batch["value_tgt"].to(self.device, non_blocking=True)
                            policy_tgt = batch["policy_tgt"].to(self.device, non_blocking=True)
                            n_obs = len(state_tensors) # Total number of obs in this batch
                            n_obs_total += n_obs # Aggregate the total nobs seen

                            if self.amp_dtype is not None:
                                with torch.autocast(device_type=self.device, dtype=self.amp_dtype):
                                    policy_logits, value_est = self.model(state_tensors)
                                    if mask_illegal_moves:  # If True, mask out illegal moves from the policy
                                        # logits with -np.inf so that the model does not get penalized for
                                        # giving them prob mass
                                        mask = self._get_legal_move_mask(batch["fen_states"],
                                                                         self.move_uci_to_idx).to(self.device)
                                        policy_logits = policy_logits.masked_fill(~mask, float('-inf'))
                                    policy_loss += policy_loss_fn(policy_logits, policy_tgt).item() * n_obs
                                    value_loss += value_loss_fn(value_est, value_tgt).item() * n_obs
                            else:
                                policy_logits, value_est = self.model(state_tensors)
                                if mask_illegal_moves:  # If True, mask out illegal moves from the policy
                                    # logits with -np.inf so that the model does not get penalized for giving
                                    # them prob mass
                                    mask = self._get_legal_move_mask(batch["fen_states"],
                                                                     self.move_uci_to_idx).to(self.device)
                                    policy_logits = policy_logits.masked_fill(~mask, float('-inf'))
                                policy_loss += policy_loss_fn(policy_logits, policy_tgt).item() * n_obs
                                value_loss += value_loss_fn(value_est, value_tgt).item() * n_obs

                        # Normalize the sum of policy and value losses by total obs in the validation set
                        policy_loss, value_loss= policy_loss / n_obs_total, value_loss / n_obs_total
                        total_loss = policy_loss + value_loss * lambda_val
                        self.val_losses.append((self.step, policy_loss, value_loss, total_loss))

                    self.model.train() # Set model back to train mode

                # Periodically save the model weights to disk, always on the last iter too
                if self.step % self.save_every == 0 or self.step == self.train_num_steps:
                    self.save(self.step)
                    # Clear the list of losses after each save, store only the ones from the last save to
                    # the next save
                    self.train_losses, self.val_losses = [], []
                    torch.cuda.empty_cache()

                del policy_logits, value_est, policy_loss, value_loss, total_loss
                del state_tensors, value_tgt, policy_tgt
                pbar.update(1)


def run_pretraining(config_name: str) -> None:
    """
    This function runs supervised pre-training for a given model config name.

    :param config_name: The name of the config to run supervised pre-training for.
    :returns: None.
    """
    # 1). Read in the config file specified by the user to be used for model training
    config = read_yaml(os.path.join(PARENT_DIR, f"config/{config_name}.yml"))

    # 2). Initialize the model specified by the config file
    model_class_dict = {"MLP": MLP, "CNN": CNN, "Transformer": Transformer}
    model_class = model_class_dict[config["model_class"]]
    model = model_class(config)  # Init the model using the config file

    # 3). Define other inputs required for training
    train_dataset_path = os.path.join(CURRENT_DIR, "lichess_train.parquet")
    train_dataloader = get_dataloader(batch_size=config['pretrain']['batch_size'],
                                      dataset_path=train_dataset_path,
                                      state_to_model_input=model.state_to_model_input)

    val_dataset_path = os.path.join(CURRENT_DIR, "lichess_val.parquet")
    val_dataloader = get_dataloader(batch_size=config['pretrain']['batch_size'],
                                    dataset_path=val_dataset_path,
                                    state_to_model_input=model.state_to_model_input)

    # 4). Train the model using the Trainer class defined above
    trainer = Trainer(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                      results_folder=config["output"]["pretrain_path"], **config["pretrain"])
    trainer.train()


if __name__ == "__main__":
    # Get the input config specified by the user for which value network model to pre-train
    parser = argparse.ArgumentParser(description="Run the supervised pre-training loop",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="The name of the config file to be used for training.")
    args = parser.parse_args()
    run_pretraining(args.config)  # Run the supervised pre-training
