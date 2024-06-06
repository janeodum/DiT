from glob import glob
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import os
from models import DiT_models
from diffusion import create_time_series_diffusion

# Define your dataset class
class TimeSeriesDataset(Dataset):
    """
    Custom dataset for loading time series data.
    """
    def __init__(self, data, labels, seq_length, patch_size):
        self.data = data
        self.labels = labels
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.num_patches = self.seq_length // self.patch_size
    
    def __len__(self):
        return len(self.data) - self.seq_length + 1
    
    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_length]
        label = self.labels[idx + self.seq_length - 1]  # use the label at the end of the sequence
        # Flatten the sequence to (num_patches, patch_size * num_features)
        seq = seq.values.reshape(self.seq_length, -1)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Define the update_ema function
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

# Define the requires_grad function
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

# Define the create_logger function
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

# Define the main function
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your CUDA installation and ensure a GPU is available.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    data = pd.read_csv(args.data_path)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    label_column = data.columns[-1]
    labels = data[label_column].values
    data = data.drop(columns=[label_column])
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    dataset = TimeSeriesDataset(data_scaled, labels, seq_length=args.seq_length, patch_size=args.patch_size)
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} time series samples ({args.data_path})")

    # Create the model
    model = DiT_models[args.model](
        seq_length=args.seq_length,
        in_channels=data_scaled.shape[1]    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = model.to(device)

    # Create the diffusion model for time series
    diffusion = create_time_series_diffusion(num_timesteps=1000, loss_type='mse')
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    accumulation_steps = 4  # Number of batches to accumulate gradients

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for step, (x, y) in enumerate(loader):
            print(f"Input x shape in train: {x.shape}")  # Debugging line
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict()
            model_output = model(x, t, **model_kwargs)
            print(f"Model output shape in train: {model_output.shape}")
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            loss = loss / accumulation_steps
            loss.backward()
            if (step + 1) % accumulation_steps == 0:
                opt.step()
                opt.zero_grad()
                update_ema(ema, model)

                running_loss += loss.item()
                log_steps += 1
                train_steps += 1
                if train_steps % args.log_every == 0:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    avg_loss = running_loss / log_steps
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=100)
    args = parser.parse_args()
    main(args)
