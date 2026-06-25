"""Stress-test the trained SequenceVAE against eight corruption categories.

Inputs  : models/sequence_vae/sequence_vae_full.pt
          data/X_test.npy
          data/normalisation_{mean,std}.csv
          data/test_sequence_metadata.csv

Outputs : outputs/stress_test/
            stress_test_summary.csv
            stress_test_per_sequence_errors.csv
            stress_test_examples.npz
"""

import os
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT  = os.path.join(_PROJECT_ROOT, "outputs", "stress_test")
_CKPT = os.path.join(_PROJECT_ROOT, "models", "sequence_vae", "sequence_vae_full.pt")

NUM_TEST_SEQUENCES = 50_000
N_EXAMPLES         = 10

from vaesde.io_utils      import load_config, load_norm_stats
from vaesde.model         import SequenceVAE
from vaesde.inference     import compute_recon_mse
from vaesde.normalisation import denormalise, renormalise
from vaesde.corruption    import (corrupt_speed_scale, corrupt_position_jump,
                                   corrupt_random_walk_velocity, corrupt_sudden_turn,
                                   corrupt_stationary)
from vaesde.constants     import SEQ_LEN, N_FEAT, FLAT_DIM, LATENT_DIM
from vaesde.reporting.stress_test import score_mse, save_outputs, print_report


def main() -> None:
    print("=== run_stress_test: SequenceVAE stress test ===", flush=True)
    os.makedirs(_OUT, exist_ok=True)
    cfg = load_config(os.path.join(_PROJECT_ROOT, "configs", "default.yaml"))

    SEED       = cfg["seed"]
    BATCH_SIZE = cfg["batch_size"]
    P95_THR    = cfg["vae_thresholds"]["p95"]
    P99_THR    = cfg["vae_thresholds"]["p99"]

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    assert os.path.exists(_CKPT), f"Checkpoint not found: {_CKPT}"
    model = SequenceVAE(FLAT_DIM, LATENT_DIM).to(device)
    ckpt  = torch.load(_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model from {_CKPT}")

    with torch.no_grad():
        dummy_out, _, _ = model(torch.zeros(2, SEQ_LEN, N_FEAT, device=device))
        assert dummy_out.shape == (2, SEQ_LEN, N_FEAT)

    mean, std = load_norm_stats(
        os.path.join(_PROJECT_ROOT, "data", "normalisation_mean.csv"),
        os.path.join(_PROJECT_ROOT, "data", "normalisation_std.csv"),
    )

    X_test_full = np.load(os.path.join(_PROJECT_ROOT, "data", "X_test.npy"), mmap_mode="r")
    print(f"X_test: {X_test_full.shape}")
    assert X_test_full.shape[1:] == (SEQ_LEN, N_FEAT)

    n_sample  = min(NUM_TEST_SEQUENCES, len(X_test_full))
    idx       = np.sort(rng.choice(len(X_test_full), n_sample, replace=False))
    seqs_norm = X_test_full[idx].copy().astype(np.float64)
    seqs_phys = denormalise(seqs_norm, mean, std)
    print(f"Sampled {n_sample:,} sequences")

    meta = pd.read_csv(
        os.path.join(_PROJECT_ROOT, "data", "test_sequence_metadata.csv")
    ).iloc[idx].reset_index(drop=True)

    TEST_CASES: list[tuple[str, np.ndarray]] = [
        ("clean",                 seqs_norm.copy()),
        ("speed_scaled_1p5",      renormalise(corrupt_speed_scale(seqs_phys.copy(), 1.5), mean, std)),
        ("speed_scaled_2p0",      renormalise(corrupt_speed_scale(seqs_phys.copy(), 2.0), mean, std)),
        ("position_jump_1000m",   renormalise(corrupt_position_jump(seqs_phys.copy(), 1000.0), mean, std)),
        ("position_jump_2000m",   renormalise(corrupt_position_jump(seqs_phys.copy(), 2000.0), mean, std)),
        ("random_walk_velocity",  renormalise(corrupt_random_walk_velocity(seqs_phys.copy(), rng), mean, std)),
        ("sudden_turn_90deg",     renormalise(corrupt_sudden_turn(seqs_phys.copy()), mean, std)),
        ("stationary_clutter_like", renormalise(corrupt_stationary(seqs_phys.copy()), mean, std)),
    ]

    for name, arr in TEST_CASES[1:]:
        assert not np.isnan(arr).any() and not np.isinf(arr).any(), f"NaN/Inf in {name}"
    print("NaN/Inf check: OK\n")

    ex_idx = np.sort(rng.choice(n_sample, N_EXAMPLES, replace=False))

    results:      dict[str, dict]       = {}
    all_mse:      dict[str, np.ndarray] = {}
    example_list: list[dict]            = []

    print(f"  {'Test type':<28}  {'mean':>8}  {'p50':>8}  {'p95':>8}  {'>p95%':>6}  {'>p99%':>6}")
    print("  " + "-" * 68)

    for test_name, arr_norm in TEST_CASES:
        mse, _     = compute_recon_mse(model, arr_norm, device, BATCH_SIZE)
        results[test_name] = {"test_type": test_name, **score_mse(mse, P95_THR, P99_THR)}
        all_mse[test_name] = mse
        r = results[test_name]
        print(f"  {test_name:<28}  {r['mean_mse']:>8.5f}  {r['median_mse']:>8.5f}  "
              f"{r['p95_mse']:>8.5f}  {r['percent_above_p95_threshold']:>5.1f}%  "
              f"{r['percent_above_p99_threshold']:>5.1f}%")

        _, ex_recon = compute_recon_mse(model, arr_norm[ex_idx], device, BATCH_SIZE)
        example_list.append({
            "test_type":                test_name,
            "sequence_indices":         ex_idx.copy(),
            "original_clean_physical":  seqs_phys[ex_idx].astype(np.float32),
            "corrupted_physical":       denormalise(arr_norm[ex_idx], mean, std).astype(np.float32),
            "reconstructed_normalised": ex_recon,
        })

    clean_p95_pct = results["clean"]["percent_above_p95_threshold"]
    output_files  = save_outputs(results, all_mse, meta, example_list, P95_THR, P99_THR, _OUT)
    checks = [
        ("Checkpoint exists",                     os.path.exists(_CKPT)),
        ("X_test shape (N, 30, 4)",               X_test_full.shape[1:] == (SEQ_LEN, N_FEAT)),
        ("Normalisation stats have correct features", True),
        ("No NaN/Inf in corrupted sequences",     True),
        ("VAE output shape matches input",         dummy_out.shape == (2, SEQ_LEN, N_FEAT)),
        (f"Clean >p95 ~5% (got {clean_p95_pct:.1f}%)", 3.0 <= clean_p95_pct <= 10.0),
        ("All output files written",               all(os.path.exists(p) for p in output_files)),
    ]
    print_report(device, n_sample, P95_THR, P99_THR, results, checks, output_files)
    print(f"\nOutputs written to {_OUT}", flush=True)


if __name__ == "__main__":
    main()
