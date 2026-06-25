"""VAE inference helpers: batched reconstruction and MSE computation."""

import numpy as np
import torch
from vaesde.model import SequenceVAE


def compute_recon_mse(model: SequenceVAE,
                      seqs_norm: np.ndarray,
                      device: torch.device,
                      batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-sequence MSE shape (N,), reconstructions shape (N, 30, 4))."""
    model.eval()
    mse_chunks, recon_chunks = [], []
    with torch.no_grad():
        for s in range(0, len(seqs_norm), batch_size):
            b = torch.from_numpy(seqs_norm[s:s + batch_size].astype(np.float32)).to(device)
            recon, _, _ = model(b)
            mse_chunks.append(((recon - b) ** 2).mean(dim=(1, 2)).cpu().numpy())
            recon_chunks.append(recon.cpu().numpy())
    return np.concatenate(mse_chunks), np.concatenate(recon_chunks, axis=0)
