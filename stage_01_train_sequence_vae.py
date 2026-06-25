"""
stage_01_train_sequence_vae.py

Trains a simple sequence VAE on fixed-length ENU trajectory windows.

Input arrays:
  data/X_train.npy   shape (1_412_436, 30, 4)
  data/X_test.npy    shape (160_946,   30, 4)

Features: [E_m, N_m, vE_mps, vN_mps]  (normalised)

Outputs written to models/sequence_vae/
"""

import csv
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEBUG_MODE             = False
DEBUG_TRAIN_SEQUENCES  = 100_000
DEBUG_TEST_SEQUENCES   = 20_000

LATENT_DIM    = 16
EPOCHS        = 30
LR            = 1e-3
BETA          = 0.001
NUM_WORKERS   = 0
SEED          = 42

SEQ_LEN  = 30
N_FEAT   = 4
FLAT_DIM = SEQ_LEN * N_FEAT   # 120

OUT_DIR = Path("models/sequence_vae")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024 if device.type == "cuda" else 256
PIN_MEMORY = device.type == "cuda"
print(f"Device : {device}  |  batch_size={BATCH_SIZE}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SequenceDataset(Dataset):
    """Wraps a memory-mapped numpy array (or a contiguous subset)."""

    def __init__(self, arr: np.ndarray):
        # arr shape: (N, 30, 4) — may be mmap or in-RAM
        self.arr = arr

    def __len__(self) -> int:
        return len(self.arr)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.arr[idx].astype(np.float32)   # copy out of mmap
        return torch.from_numpy(x)             # (30, 4)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SequenceVAE(nn.Module):

    def __init__(self, flat_dim: int = FLAT_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, flat_dim),
        )

    def encode(self, x: torch.Tensor):
        h      = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        # x : (batch, 30, 4)
        batch = x.size(0)
        x_flat = x.view(batch, FLAT_DIM)          # (batch, 120)
        mu, logvar = self.encode(x_flat)
        z          = self.reparameterise(mu, logvar)
        recon_flat = self.decode(z)               # (batch, 120)
        recon      = recon_flat.view(batch, SEQ_LEN, N_FEAT)
        return recon, mu, logvar


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def vae_loss(recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = BETA):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total      = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------
def run_epoch(model: SequenceVAE, loader: DataLoader,
              optimiser=None) -> tuple[float, float, float]:
    """One pass over loader.  If optimiser is None → eval mode."""
    training = optimiser is not None
    model.train(training)
    tot_loss = tot_recon = tot_kl = 0.0

    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device, non_blocking=PIN_MEMORY)   # (B, 30, 4)
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar)

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            n = batch.size(0)
            tot_loss  += loss.item()  * n
            tot_recon += recon_loss.item() * n
            tot_kl    += kl_loss.item()    * n

    N = len(loader.dataset)
    return tot_loss / N, tot_recon / N, tot_kl / N


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load arrays (memory-mapped)
    # -----------------------------------------------------------------------
    print("Memory-mapping arrays ...")
    X_train_full = np.load("data/X_train.npy", mmap_mode="r")
    X_test_full  = np.load("data/X_test.npy",  mmap_mode="r")
    print(f"  X_train full shape : {X_train_full.shape}")
    print(f"  X_test  full shape : {X_test_full.shape}")

    # Sanity: shape compatibility
    assert X_train_full.ndim == 3 and X_train_full.shape[1:] == (SEQ_LEN, N_FEAT), \
        f"X_train shape mismatch: {X_train_full.shape}"
    assert X_test_full.ndim == 3  and X_test_full.shape[1:]  == (SEQ_LEN, N_FEAT), \
        f"X_test shape mismatch: {X_test_full.shape}"

    # -----------------------------------------------------------------------
    # Debug subset
    # -----------------------------------------------------------------------
    meta_test_full = pd.read_csv("data/test_sequence_metadata.csv")

    if DEBUG_MODE:
        n_tr = min(DEBUG_TRAIN_SEQUENCES, len(X_train_full))
        n_te = min(DEBUG_TEST_SEQUENCES,  len(X_test_full))
        train_idx = np.random.choice(len(X_train_full), n_tr, replace=False)
        test_idx  = np.random.choice(len(X_test_full),  n_te, replace=False)
        train_idx.sort()
        test_idx.sort()
        # Copy debug subsets into RAM (small enough)
        X_train = X_train_full[train_idx].copy()
        X_test  = X_test_full[test_idx].copy()
        meta_test = meta_test_full.iloc[test_idx].reset_index(drop=True)
        suffix = "debug"
        print(f"\nDEBUG MODE: using {len(X_train):,} train, {len(X_test):,} test sequences")
    else:
        # For full mode, keep arrays as mmap views; Dataset will copy per-sample
        X_train   = X_train_full
        X_test    = X_test_full
        meta_test = meta_test_full
        suffix    = "full"
        print(f"\nFULL MODE: using {len(X_train):,} train, {len(X_test):,} test sequences")

    # Sanity: NaN / Inf check on a small batch
    check = X_train[:min(5000, len(X_train))].astype(np.float32)
    assert not np.isnan(check).any(),  "NaN found in sampled training data"
    assert not np.isinf(check).any(),  "Inf found in sampled training data"
    print("  NaN/Inf check on sampled train batch: OK")

    # -----------------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------------
    train_ds     = SequenceDataset(X_train)
    test_ds      = SequenceDataset(X_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model     = SequenceVAE(FLAT_DIM, LATENT_DIM).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    # Sanity: output shape
    with torch.no_grad():
        dummy     = torch.zeros(2, SEQ_LEN, N_FEAT, device=device)
        dummy_out, _, _ = model(dummy)
        assert dummy_out.shape == (2, SEQ_LEN, N_FEAT), \
            f"Model output shape mismatch: {dummy_out.shape}"
    print("  Model output shape check: OK")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    history: list[dict] = []
    print()
    print(f"{'Epoch':>5}  {'Tr Total':>10}  {'Tr Recon':>10}  {'Tr KL':>10}"
          f"  {'Te Total':>10}  {'Te Recon':>10}  {'Te KL':>10}  {'Time':>6}")
    print("-" * 80)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        tr_total, tr_recon, tr_kl = run_epoch(model, train_loader, optimiser)
        te_total, te_recon, te_kl = run_epoch(model, test_loader)
        elapsed = time.time() - t_ep

        history.append({
            "epoch":        epoch,
            "tr_total":     tr_total,
            "tr_recon":     tr_recon,
            "tr_kl":        tr_kl,
            "te_total":     te_total,
            "te_recon":     te_recon,
            "te_kl":        te_kl,
        })
        print(f"{epoch:>5}  {tr_total:>10.6f}  {tr_recon:>10.6f}  {tr_kl:>10.6f}"
              f"  {te_total:>10.6f}  {te_recon:>10.6f}  {te_kl:>10.6f}  {elapsed:>5.1f}s")

    total_time = time.time() - t0
    print(f"\nTotal training time: {total_time:.1f}s")

    # Sanity: loss decreases
    first_loss = history[0]["tr_total"]
    last_loss  = history[-1]["tr_total"]
    loss_decreased = last_loss < first_loss
    print(f"Loss decreased (epoch 1 → {EPOCHS}): {first_loss:.6f} → {last_loss:.6f}  "
          f"{'OK' if loss_decreased else 'WARNING: did not decrease'}")

    # -----------------------------------------------------------------------
    # Save model checkpoint
    # -----------------------------------------------------------------------
    ckpt_path = OUT_DIR / f"sequence_vae_{suffix}.pt"
    torch.save({
        "epoch":       EPOCHS,
        "model_state": model.state_dict(),
        "optim_state": optimiser.state_dict(),
        "config": {
            "latent_dim":    LATENT_DIM,
            "flat_dim":      FLAT_DIM,
            "seq_len":       SEQ_LEN,
            "n_feat":        N_FEAT,
            "beta":          BETA,
            "lr":            LR,
            "batch_size":    BATCH_SIZE,
            "debug_mode":    DEBUG_MODE,
        },
    }, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    # -----------------------------------------------------------------------
    # Save loss history
    # -----------------------------------------------------------------------
    loss_path = OUT_DIR / "loss_history.csv"
    pd.DataFrame(history).to_csv(loss_path, index=False)

    # -----------------------------------------------------------------------
    # Per-sequence reconstruction MSE on test set
    # -----------------------------------------------------------------------
    print("\nComputing per-sequence reconstruction MSE on test set ...")
    model.eval()
    mse_list: list[float] = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=PIN_MEMORY)
            recon, _, _ = model(batch)
            # per-sequence MSE: mean over (30, 4) dimensions
            mse = ((recon - batch) ** 2).mean(dim=(1, 2))   # (B,)
            mse_list.extend(mse.cpu().numpy().tolist())

    mse_arr = np.array(mse_list)

    # Build test reconstruction errors CSV
    errors_df = meta_test[["sequence_id", "segment_id", "start_time", "end_time"]].copy()
    errors_df["recon_mse"] = mse_arr
    errors_path = OUT_DIR / "test_reconstruction_errors.csv"
    errors_df.to_csv(errors_path, index=False)

    # -----------------------------------------------------------------------
    # Reconstruction error summary
    # -----------------------------------------------------------------------
    summary_df = pd.DataFrame([{
        "split":                   "test",
        "num_sequences_evaluated": len(mse_arr),
        "mean_recon_mse":          float(np.mean(mse_arr)),
        "median_recon_mse":        float(np.median(mse_arr)),
        "p90_recon_mse":           float(np.percentile(mse_arr, 90)),
        "p95_recon_mse":           float(np.percentile(mse_arr, 95)),
        "p99_recon_mse":           float(np.percentile(mse_arr, 99)),
        "max_recon_mse":           float(np.max(mse_arr)),
    }])
    summary_path = OUT_DIR / "reconstruction_error_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # -----------------------------------------------------------------------
    # Example reconstructions (20 random test sequences)
    # -----------------------------------------------------------------------
    n_examples = min(20, len(X_test))
    example_idx = np.random.choice(len(X_test), n_examples, replace=False)
    example_idx.sort()
    example_batch = torch.from_numpy(
        X_test[example_idx].astype(np.float32)
    ).to(device)
    with torch.no_grad():
        example_recon, _, _ = model(example_batch)
    np.savez(
        OUT_DIR / "example_reconstructions.npz",
        original         = X_test[example_idx],
        reconstructed    = example_recon.cpu().numpy(),
        sequence_indices = example_idx,
    )

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    print()
    print("=" * 62)
    print("TRAINING COMPLETE")
    print("=" * 62)
    print(f"  Device used                   : {device}")
    print(f"  Debug mode                    : {DEBUG_MODE}")
    print(f"  Final train total loss        : {history[-1]['tr_total']:.6f}")
    print(f"  Final train recon loss        : {history[-1]['tr_recon']:.6f}")
    print(f"  Final train KL loss           : {history[-1]['tr_kl']:.6f}")
    print(f"  Final test  total loss        : {history[-1]['te_total']:.6f}")
    print(f"  Final test  recon loss        : {history[-1]['te_recon']:.6f}")
    print(f"  Final test  KL loss           : {history[-1]['te_kl']:.6f}")
    print()
    print(f"  Recon MSE (test) p50          : {np.percentile(mse_arr, 50):.6f}")
    print(f"  Recon MSE (test) p90          : {np.percentile(mse_arr, 90):.6f}")
    print(f"  Recon MSE (test) p95          : {np.percentile(mse_arr, 95):.6f}")
    print(f"  Recon MSE (test) p99          : {np.percentile(mse_arr, 99):.6f}")
    print()
    print("Sanity checks:")
    checks = [
        ("X_train shape compatible (N, 30, 4)",
            X_train_full.shape[1:] == (SEQ_LEN, N_FEAT)),
        ("X_test  shape compatible (N, 30, 4)",
            X_test_full.shape[1:]  == (SEQ_LEN, N_FEAT)),
        ("No NaN/Inf in sampled train batch",
            True),   # checked above; would have asserted otherwise
        ("Model output shape == input shape",
            dummy_out.shape == (2, SEQ_LEN, N_FEAT)),
        ("Loss decreased over training",
            loss_decreased),
        ("reconstruction_error_summary.csv written",
            summary_path.exists()),
        ("Model checkpoint written",
            ckpt_path.exists()),
    ]
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")


if __name__ == "__main__":
    main()
