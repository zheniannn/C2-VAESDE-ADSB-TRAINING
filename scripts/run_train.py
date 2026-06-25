"""Train a SequenceVAE on ENU trajectory windows.

Inputs  : data/X_train.npy  (N, 30, 4)
          data/X_test.npy   (N, 30, 4)
          data/test_sequence_metadata.csv

Outputs : outputs/train/
            sequence_vae_{debug|full}.pt
            loss_history.csv
            test_reconstruction_errors.csv
            reconstruction_error_summary.csv
            example_reconstructions.npz
"""

import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT    = os.path.join(_PROJECT_ROOT, "outputs", "train")
_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "default.yaml")

N_EXAMPLES = 20

from vaesde.io_utils   import load_config
from vaesde.model      import SequenceVAE
from vaesde.training   import SequenceDataset, run_epoch
from vaesde.constants  import SEQ_LEN, N_FEAT, FLAT_DIM, LATENT_DIM
from vaesde.reporting  import train as rpt


def main() -> None:
    print("=== run_train: SequenceVAE training ===", flush=True)
    os.makedirs(_OUT, exist_ok=True)
    cfg = load_config(_CONFIG)

    SEED        = cfg["seed"]
    BATCH_SIZE  = cfg["batch_size"]
    EPOCHS      = cfg["training"]["epochs"]
    LR          = cfg["training"]["lr"]
    BETA        = cfg["training"]["beta"]
    DEBUG_MODE  = cfg["training"]["debug_mode"]
    DEBUG_TRAIN = cfg["training"]["debug_train_sequences"]
    DEBUG_TEST  = cfg["training"]["debug_test_sequences"]

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sz = BATCH_SIZE if device.type == "cuda" else 256
    pin_mem  = device.type == "cuda"
    print(f"Device : {device}  |  batch_size={batch_sz}")

    print("Memory-mapping arrays ...")
    X_train_full = np.load(os.path.join(_PROJECT_ROOT, "data", "X_train.npy"), mmap_mode="r")
    X_test_full  = np.load(os.path.join(_PROJECT_ROOT, "data", "X_test.npy"),  mmap_mode="r")
    print(f"  X_train: {X_train_full.shape}  X_test: {X_test_full.shape}")
    assert X_train_full.shape[1:] == (SEQ_LEN, N_FEAT)
    assert X_test_full.shape[1:]  == (SEQ_LEN, N_FEAT)

    meta_test_full = pd.read_csv(os.path.join(_PROJECT_ROOT, "data", "test_sequence_metadata.csv"))

    if DEBUG_MODE:
        n_tr = min(DEBUG_TRAIN, len(X_train_full))
        n_te = min(DEBUG_TEST,  len(X_test_full))
        tr_idx = np.sort(np.random.choice(len(X_train_full), n_tr, replace=False))
        te_idx = np.sort(np.random.choice(len(X_test_full),  n_te, replace=False))
        X_train   = X_train_full[tr_idx].copy()
        X_test    = X_test_full[te_idx].copy()
        meta_test = meta_test_full.iloc[te_idx].reset_index(drop=True)
        suffix    = "debug"
        print(f"\nDEBUG: {len(X_train):,} train / {len(X_test):,} test")
    else:
        X_train   = X_train_full
        X_test    = X_test_full
        meta_test = meta_test_full
        suffix    = "full"
        print(f"\nFULL: {len(X_train):,} train / {len(X_test):,} test")

    check = X_train[:min(5000, len(X_train))].astype(np.float32)
    assert not np.isnan(check).any() and not np.isinf(check).any()
    print("  NaN/Inf check: OK")

    train_loader = DataLoader(SequenceDataset(X_train), batch_size=batch_sz,
                              shuffle=True,  num_workers=0, pin_memory=pin_mem)
    test_loader  = DataLoader(SequenceDataset(X_test),  batch_size=batch_sz,
                              shuffle=False, num_workers=0, pin_memory=pin_mem)

    model     = SequenceVAE(FLAT_DIM, LATENT_DIM).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    with torch.no_grad():
        dummy     = torch.zeros(2, SEQ_LEN, N_FEAT, device=device)
        dummy_out, _, _ = model(dummy)
        assert dummy_out.shape == (2, SEQ_LEN, N_FEAT)
    print(f"  Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    history: list[dict] = []
    print(f"\n{'Epoch':>5}  {'Tr Total':>10}  {'Tr Recon':>10}  {'Tr KL':>10}"
          f"  {'Te Total':>10}  {'Te Recon':>10}  {'Te KL':>10}  {'Time':>6}")
    print("-" * 80)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        tr_total, tr_recon, tr_kl = run_epoch(model, train_loader, device, BETA, optimiser)
        te_total, te_recon, te_kl = run_epoch(model, test_loader,  device, BETA)
        elapsed = time.time() - t_ep
        history.append({"epoch": epoch, "tr_total": tr_total, "tr_recon": tr_recon,
                         "tr_kl": tr_kl, "te_total": te_total, "te_recon": te_recon,
                         "te_kl": te_kl})
        print(f"{epoch:>5}  {tr_total:>10.6f}  {tr_recon:>10.6f}  {tr_kl:>10.6f}"
              f"  {te_total:>10.6f}  {te_recon:>10.6f}  {te_kl:>10.6f}  {elapsed:>5.1f}s")
    print(f"\nTotal training time: {time.time() - t0:.1f}s")

    loss_decreased = history[-1]["tr_total"] < history[0]["tr_total"]

    ckpt_path = os.path.join(_OUT, f"sequence_vae_{suffix}.pt")
    torch.save({
        "epoch": EPOCHS, "model_state": model.state_dict(),
        "optim_state": optimiser.state_dict(),
        "config": {"latent_dim": LATENT_DIM, "flat_dim": FLAT_DIM,
                   "seq_len": SEQ_LEN, "n_feat": N_FEAT,
                   "beta": BETA, "lr": LR, "batch_size": batch_sz,
                   "debug_mode": DEBUG_MODE},
    }, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    model.eval()
    mse_list: list[float] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=pin_mem)
            recon, _, _ = model(batch)
            mse_list.extend(((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy().tolist())
    mse_arr = np.array(mse_list)

    ex_idx   = np.sort(np.random.choice(len(X_test), min(N_EXAMPLES, len(X_test)), replace=False))
    ex_batch = torch.from_numpy(X_test[ex_idx].astype(np.float32)).to(device)
    with torch.no_grad():
        ex_recon, _, _ = model(ex_batch)

    rpt.save_loss_history(history, _OUT)
    errors_path  = rpt.save_reconstruction_errors(meta_test, mse_arr, _OUT)
    summary_path = rpt.save_reconstruction_summary(mse_arr, _OUT)
    npz_path     = rpt.save_example_reconstructions(X_test, ex_idx, ex_recon.cpu().numpy(), _OUT)

    checks = [
        ("X_train shape (N, 30, 4)",          X_train_full.shape[1:] == (SEQ_LEN, N_FEAT)),
        ("X_test  shape (N, 30, 4)",           X_test_full.shape[1:]  == (SEQ_LEN, N_FEAT)),
        ("No NaN/Inf in sampled train batch",  True),
        ("Model output shape == input shape",  dummy_out.shape == (2, SEQ_LEN, N_FEAT)),
        ("Loss decreased over training",       loss_decreased),
        ("reconstruction_error_summary written", os.path.exists(summary_path)),
        ("Checkpoint written",                 os.path.exists(ckpt_path)),
    ]
    rpt.print_training_summary(device, DEBUG_MODE, history, mse_arr, checks)
    print(f"\nOutputs written to {_OUT}", flush=True)


if __name__ == "__main__":
    main()
