"""Dataset, loss, and epoch helpers for training the SequenceVAE."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from vaesde.model import SequenceVAE


class SequenceDataset(Dataset):
    """Wraps a memory-mapped numpy array (or a contiguous subset)."""

    def __init__(self, arr: np.ndarray):
        self.arr = arr

    def __len__(self) -> int:
        return len(self.arr)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.arr[idx].astype(np.float32))


def vae_loss(recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO loss: MSE reconstruction + beta-weighted KL divergence."""
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def run_epoch(model: SequenceVAE,
              loader: DataLoader,
              device: torch.device,
              beta: float,
              optimiser=None) -> tuple[float, float, float]:
    """One pass over loader.  Pass optimiser=None for eval mode."""
    training = optimiser is not None
    model.train(training)
    pin = device.type == "cuda"
    tot_loss = tot_recon = tot_kl = 0.0

    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device, non_blocking=pin)
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, beta)

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            n = batch.size(0)
            tot_loss  += loss.item()       * n
            tot_recon += recon_loss.item() * n
            tot_kl    += kl_loss.item()    * n

    N = len(loader.dataset)
    return tot_loss / N, tot_recon / N, tot_kl / N
