"""
### TODO ADD HERE
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
from typing import Tuple, Callable, Dict
import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from utils.general import get_device, get_amp_dtype
from core.torch_models import MLP, CNN, Transformer
from utils.general import read_yaml

class SupervisedPretrainingDataset(Dataset):
    """
    A dataset for supervised pre-training of a value network on stockfish eval scores.
    """

    def __init__(self, spt_dataset_path: str, state_to_model_input: Callable):
        """
        Initializes a dataset object for supervised pre-training of a value network.

        :param spt_dataset_path: A file path for where the supervised pre-training parquet dataset is cached.
        :param state_to_model_input: A callable function that converts a batch of FEN board state encodings
            into a torch.Tensor that can be passed into v_network.
        """
        self.spt_dataset = pd.read_parquet(spt_dataset_path) # Read in the data as a pd.DataFrame
        self.state_to_model_input = state_to_model_input # Function for converting a FEN into a tensor

    def __len__(self):
        """
        Returns the total number of observations in the dataset.
        """
        return len(self.spt_dataset)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary containing keys "boards" and "eval_tgts" for a particular index in the dataset.
        """
        return {"boards": self.state_to_model_input([self.spt_dataset.loc[idx, "fen"]]).squeeze(0),
                "eval_tgts": torch.tensor(self.spt_dataset.loc[idx, "tgt"], dtype=torch.float32)}


def get_dataloader(batch_size: int, spt_dataset_path: str, state_to_model_input: Callable) -> DataLoader:
    """
    Creates a dataloader for supervised pre-training of a value network on stockfish eval scores.

    :param batch_size: The size of the batches to return from the dataloader.
    :param spt_dataset_path: A file path for where the supervised pre-training parquet dataset is cached.
    :param state_to_model_input: A callable function that converts a batch of FEN board state encodings
        into a torch.Tensor that can be passed into v_network.
    :returns: A pytorch DataLoader object.
    """
    device = get_device()  # Auto-detect the available hardware
    dataset = SupervisedPretrainingDataset(spt_dataset_path, state_to_model_input)
    if device == "cuda":
        num_workers, pin_memory, persistent_workers = 4, True, True
    else:
        num_workers, pin_memory, persistent_workers = 0, False, False
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=pin_memory, persistent_workers=persistent_workers)


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
    def __init__(self, model: nn.Module, dataloader: DataLoader, lr_start: float = 1e-4,
                 lr_end: float = 1e-5, weight_decay: float = 1e-3, train_num_steps: int = 300000,
                 adam_betas: Tuple[float] = (0.9, 0.99), grad_clip: float = 1.0,
                 save_every: int = 10000, results_folder: str = None, use_amp: bool = False,
                 use_latest_checkpoint: bool = True):
        """
        This is a framework for training a deep value network model. This class wrapper has methods for
        loading a model from a recent checkpoint, saving a model periodically during training, and running a
        training loop to train from scratch or to continue from the last checkpoint.

        :param model: A value network implemented in pytorch.
        :param dataloader: A dataloader that will yield the training batches.
        :param lr_start: The initial learning rate.
        :param lr_end: The terminal training learning rate.
        :param weight_decay: The weight_decay to provide to the Adam optimizer for L2 regularization.
        :param train_num_steps: The number of training steps to run in total.
        :param adam_betas: Beta parameters for the adam optimizer.
        :param grad_clip: The amount of gradient clipping to use during training.
        :param save_every: An int denoting how often to save the model weights and losses.
        :param results_folder: A location to save the results of training.
        :param use_amp: Whether to use automatic mixed-precision type casting during training.
        :param use_latest_checkpoint: If set to True, then the latest checkpoint detected in the results
            directory will be loaded in before training begins to pick up from where it was last left off.
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

        self.model = model # The value network being trained
        self.logger.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
        self.device = get_device()  # Auto-detect what device to use for training
        self.grad_clip = grad_clip  # The amount of gradient clipping to use during training
        self.amp_dtype = get_amp_dtype(self.device) if use_amp else None
        self.save_every = save_every  # The frequency of saving model weights
        self.train_num_steps = train_num_steps  # The total number of training steps

        # Save a pointer to the train and validation dataloaders
        self.dataloader = dataloader

        # Configure the optimizer for training
        self.opt = AdamW(self.model.parameters(), lr=lr_start, betas=adam_betas, weight_decay=weight_decay)

        warmup_steps = 5000 # Slowly ramp up the learning rate from very low to peak
        warmup = LinearLR(self.opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        # Linearly decay the learning rate during training
        decay = LinearLR(self.opt,start_factor=1.0, end_factor=lr_end / lr_start,
                         total_iters=train_num_steps - warmup_steps)
        # Stack both the learning rate warm up and the gradual linear decay into 1 scheduler
        self.scheduler = SequentialLR(self.opt, schedulers=[warmup, decay], milestones=[warmup_steps])

        self.step = 0  # Training step counter
        self.all_losses = []  # Aggregate loss values during training

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
        pd.Series(self.all_losses).to_csv(os.path.join(self.losses_folder, f"losses-{milestone}.csv"))

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
        self.scheduler.load_state_dict(checkpoint_data["scheduler"])
        # Losses are not loaded in, they are saved to disk periodically with the model weights and are not
        # needed to continue training. The losses obtained by training will be cached again at the next save

        # Move the model and the optimizer to the same device to continue training or for inference
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def train(self) -> None:
        """
        Runs the training of the model until completion for self.train_num_steps total training iterations.
        """
        self.logger.info(f"Starting Training, device={self.device}, amp_dtype={self.amp_dtype}")
        for i, param_group in enumerate(self.opt.param_groups):  # Report the learning rate and weight decay
            self.logger.info(f"lr={param_group['lr']}, wd={param_group['weight_decay']}")
            break # Show for only the first parameter group, assume all are the same

        loss_fn = nn.HuberLoss(delta=0.1) # Use a Huber Loss function for training, a mix between MSE and MAE

        self.model.to(self.device)  # Move the model to the correct device
        self.model.train()  # Make sure to set the model to train mode for training

        inf_dataloader = infinite_loader(self.dataloader)  # This does not cache batches

        if self.amp_dtype is not None:
            if self.device != 'cuda':
                self.logger.info("AMP with FP16 requires CUDA")
                self.amp_dtype = None
            else:
                scaler = torch.amp.GradScaler('cuda')

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:  # Run until all training iterations are complete
                # Get the next training batch and move it to the same device as the model
                batch = next(inf_dataloader)
                boards = batch["boards"].to(self.device, non_blocking=True)
                eval_tgts = batch["eval_tgts"].to(self.device, non_blocking=True)

                self.opt.zero_grad(set_to_none=True)  # Zero the grads of the opt before computing the loss
                # Compute the forward-pass through the model and compute a tensor that is the same shape
                # along the first 2 dims as captions but also gives the prob dist across the vocab
                if self.amp_dtype is not None:
                    with torch.autocast(device_type=self.device, dtype=self.amp_dtype):
                        outputs = self.model(boards).squeeze(1)
                        loss = loss_fn(outputs, eval_tgts)
                else:
                    outputs = self.model(boards).squeeze(1)
                    loss = loss_fn(outputs, eval_tgts)

                if self.amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        scaler.unscale_(self.opt)  # Unscale before clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    scaler.step(self.opt)  # Update the model parameters by taking a gradient step
                    scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                                   self.grad_clip)
                    self.opt.step()  # Update the model parameters by taking a gradient step

                pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm:.3f}")

                self.scheduler.step() # Update the learning rate scheduler

                self.all_losses.append(loss.item())  # Aggregate all the loss values for each timestep
                self.step += 1

                # Periodically save the model weights to disk
                if self.step % self.save_every == 0 or self.step == self.train_num_steps:
                    self.save(self.step)
                    self.all_losses = []  # Clear the list of losses after each save, store only the ones
                    # from the last save to the next save
                    torch.cuda.empty_cache()

                del outputs, loss, boards, eval_tgts

                pbar.update(1)

def run_pretraining(config_name: str) -> None:
    """
    This function runs supervised pre-training for a given model config name.

    :param config_name: The name of the config to run supervised pre-training for.
    :returns: None.
    """
    # 1). Read in the config file specified by the user to be used for model training
    config = read_yaml(os.path.join(PARENT_DIR, f"config/{args.config}.yml"))

    # 2). Initialize the model specified by the config file
    model_class_dict = {"MLP": MLP, "CNN": CNN, "Transformer": Transformer}
    model_class = model_class_dict[config["model_class"]]
    v_network = model_class(config)  # Init the value network using the config file
    model = v_network.model # The pytorch model on the inside

    # 3). Define other inputs required for training
    spt_dataset_path = os.path.join(CURRENT_DIR, "spt_dataset.parquet")
    dataloader = get_dataloader(batch_size=512, spt_dataset_path=spt_dataset_path,
                                state_to_model_input=v_network.state_to_model_input)

    # 4). Train the value network using the Trainer class defined above
    trainer = Trainer(model=model, dataloader=dataloader, lr_start=1e-3, lr_end=1e-4, weight_decay=1e-2,
                      train_num_steps=300000, grad_clip=1.0, save_every=10000,
                      results_folder=os.path.join(CURRENT_DIR, config["model"]), use_amp=True,
                      use_latest_checkpoint=True)
    trainer.train()

if __name__ == "__main__":
    # Get the input config specified by the user for which value network model to pre-train
    parser = argparse.ArgumentParser(description="Run the supervised pre-training loop for a value network",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="The name of the config file to be used for training.")
    args = parser.parse_args()
    run_pretraining(args.config) # Run the supervised pre-training


