"""Output helpers for the training stage — loss history, errors, examples."""

import os
import numpy as np
import pandas as pd


def save_loss_history(history: list[dict], output_dir: str) -> None:
    """Write per-epoch loss CSV."""
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "loss_history.csv"), index=False)


def save_reconstruction_errors(meta: pd.DataFrame,
                                mse_arr: np.ndarray,
                                output_dir: str) -> str:
    """Write per-sequence test reconstruction errors CSV; return path."""
    df = meta[["sequence_id", "segment_id", "start_time", "end_time"]].copy()
    df["recon_mse"] = mse_arr
    path = os.path.join(output_dir, "test_reconstruction_errors.csv")
    df.to_csv(path, index=False)
    return path


def save_reconstruction_summary(mse_arr: np.ndarray, output_dir: str) -> str:
    """Write MSE distribution summary CSV; return path."""
    summary = pd.DataFrame([{
        "split":                   "test",
        "num_sequences_evaluated": len(mse_arr),
        "mean_recon_mse":          float(np.mean(mse_arr)),
        "median_recon_mse":        float(np.median(mse_arr)),
        "p90_recon_mse":           float(np.percentile(mse_arr, 90)),
        "p95_recon_mse":           float(np.percentile(mse_arr, 95)),
        "p99_recon_mse":           float(np.percentile(mse_arr, 99)),
        "max_recon_mse":           float(np.max(mse_arr)),
    }])
    path = os.path.join(output_dir, "reconstruction_error_summary.csv")
    summary.to_csv(path, index=False)
    return path


def save_example_reconstructions(X_test: np.ndarray,
                                  example_idx: np.ndarray,
                                  example_recon: np.ndarray,
                                  output_dir: str) -> str:
    """Save original/reconstructed example sequences as NPZ; return path."""
    path = os.path.join(output_dir, "example_reconstructions.npz")
    np.savez(
        path,
        original         = X_test[example_idx],
        reconstructed    = example_recon,
        sequence_indices = example_idx,
    )
    return path


def print_training_summary(device, debug_mode: bool, history: list[dict],
                            mse_arr: np.ndarray, checks: list[tuple]) -> None:
    """Print final training report to stdout."""
    print()
    print("=" * 62)
    print("TRAINING COMPLETE")
    print("=" * 62)
    print(f"  Device used                   : {device}")
    print(f"  Debug mode                    : {debug_mode}")
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
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
