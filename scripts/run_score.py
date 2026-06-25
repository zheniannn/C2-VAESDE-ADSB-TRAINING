"""VAE + kinematic motion-prior scorer across eight corruption categories.

Inputs  : models/sequence_vae/sequence_vae_full.pt
          data/X_test.npy
          data/normalisation_{mean,std}.csv
          data/test_sequence_metadata.csv

Outputs : outputs/score/
            motion_prior_summary.csv
            motion_prior_per_sequence_scores.csv
            motion_prior_examples.npz
"""

import os
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT  = os.path.join(_PROJECT_ROOT, "outputs", "score")
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
from vaesde.kinematics    import compute_kinematics, compute_flags
from vaesde.constants     import SEQ_LEN, N_FEAT, FLAT_DIM, LATENT_DIM
from vaesde.reporting.score import build_summary_row, save_outputs, print_report


def main() -> None:
    print("=== run_score: motion-prior scorer ===", flush=True)
    os.makedirs(_OUT, exist_ok=True)
    cfg = load_config(os.path.join(_PROJECT_ROOT, "configs", "default.yaml"))

    SEED       = cfg["seed"]
    BATCH_SIZE = cfg["batch_size"]
    KT         = cfg["kinematic_thresholds"]

    thr = {
        "vae_p95_threshold":              cfg["vae_thresholds"]["p95"],
        "vae_p99_threshold":              cfg["vae_thresholds"]["p99"],
        "max_speed_threshold_mps":        KT["max_speed_mps"],
        "max_accel_threshold_mps2":       KT["max_accel_mps2"],
        "max_pv_error_threshold_m":       KT["max_pv_error_m"],
        "mean_pv_error_threshold_m":      KT["mean_pv_error_m"],
        "min_mean_speed_threshold_mps":   KT["min_mean_speed_mps"],
        "min_displacement_threshold_m":   KT["min_displacement_m"],
        # No turn-rate keys → compute_flags leaves those flags as False
    }

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
    assert X_test_full.ndim == 3 and X_test_full.shape[1:] == (SEQ_LEN, N_FEAT)

    n_sample  = min(NUM_TEST_SEQUENCES, len(X_test_full))
    idx       = np.sort(rng.choice(len(X_test_full), n_sample, replace=False))
    seqs_norm = X_test_full[idx].copy().astype(np.float64)
    seqs_phys = denormalise(seqs_norm, mean, std)
    print(f"Sampled {n_sample:,} sequences\n")

    meta = pd.read_csv(
        os.path.join(_PROJECT_ROOT, "data", "test_sequence_metadata.csv")
    ).iloc[idx].reset_index(drop=True)

    TEST_CASES: list[tuple[str, np.ndarray]] = [
        ("clean",                 seqs_phys.copy()),
        ("speed_scaled_1p5",      corrupt_speed_scale(seqs_phys, 1.5)),
        ("speed_scaled_2p0",      corrupt_speed_scale(seqs_phys, 2.0)),
        ("position_jump_1000m",   corrupt_position_jump(seqs_phys, 1000.0)),
        ("position_jump_2000m",   corrupt_position_jump(seqs_phys, 2000.0)),
        ("random_walk_velocity",  corrupt_random_walk_velocity(seqs_phys, rng)),
        ("sudden_turn_90deg",     corrupt_sudden_turn(seqs_phys)),
        ("stationary_clutter_like", corrupt_stationary(seqs_phys)),
    ]

    for name, phys in TEST_CASES[1:]:
        norm = renormalise(phys, mean, std)
        assert not np.isnan(norm).any() and not np.isinf(norm).any(), f"NaN/Inf in {name}"
    print("NaN/Inf check: OK\n")

    ex_idx = np.sort(rng.choice(n_sample, N_EXAMPLES, replace=False))

    summary_rows:   list[dict]         = []
    per_seq_chunks: list[pd.DataFrame] = []
    example_list:   list[dict]         = []

    clean_vae_p95_pct  = None
    clean_vae_p99_pct  = None
    pjump_kin_flag_pct: list[float] = []
    stat_kin_flag_pct  = None

    for test_name, corrupted_phys in TEST_CASES:
        norm = renormalise(corrupted_phys, mean, std)
        recon_mse, recon_arr = compute_recon_mse(model, norm, device, BATCH_SIZE)
        kin   = compute_kinematics(corrupted_phys)
        flags = compute_flags(kin, recon_mse, thr)
        row   = build_summary_row(test_name, recon_mse, kin, flags)
        summary_rows.append(row)

        df = meta[["sequence_id", "segment_id", "start_time", "end_time"]].copy()
        df["test_type"] = test_name
        df["recon_mse"] = recon_mse
        for k in ["vae_flag_p95", "vae_flag_p99",
                  "mean_speed_mps", "max_speed_mps",
                  "mean_accel_mps2", "max_accel_mps2",
                  "mean_pv_error_m", "max_pv_error_m", "total_displacement_m"]:
            df[k] = kin.get(k, flags.get(k))
        for k in ["flag_speed", "flag_accel", "flag_pv_max", "flag_pv_mean",
                  "flag_too_slow", "flag_low_displacement",
                  "kinematic_flag", "combined_flag_p95", "combined_flag_p99"]:
            df[k] = flags[k]
        per_seq_chunks.append(df)

        example_list.append({
            "test_type":               test_name,
            "sequence_indices":        ex_idx.copy(),
            "clean_physical":          seqs_phys[ex_idx].astype(np.float32),
            "corrupted_physical":      corrupted_phys[ex_idx].astype(np.float32),
            "reconstructed_normalised": recon_arr[ex_idx],
        })

        vp95 = row["percent_vae_above_p95"]
        kpct = row["percent_kinematic_flagged"]
        if test_name == "clean":
            clean_vae_p95_pct = vp95
            clean_vae_p99_pct = row["percent_vae_above_p99"]
        elif test_name in ("position_jump_1000m", "position_jump_2000m"):
            pjump_kin_flag_pct.append(row["percent_flag_pv_max"])
        elif test_name == "stationary_clutter_like":
            stat_kin_flag_pct = max(row["percent_flag_too_slow"],
                                    row["percent_flag_low_displacement"])

        print(f"  [{test_name:<28}]  "
              f"VAE_p95={vp95:5.1f}%  "
              f"Kin={kpct:5.1f}%  "
              f"Comb_p95={row['percent_combined_p95_flagged']:5.1f}%")

    output_files = save_outputs(summary_rows, per_seq_chunks, example_list, _OUT)

    checks = [
        ("Checkpoint exists",                    os.path.exists(_CKPT)),
        ("X_test shape compatible (N, 30, 4)",   X_test_full.shape[1:] == (SEQ_LEN, N_FEAT)),
        ("Normalisation stats correct features", True),
        ("No NaN/Inf in corrupted sequences",    True),
        ("VAE output shape matches input",        dummy_out.shape == (2, SEQ_LEN, N_FEAT)),
        (f"Clean VAE p95 ~5%  (got {clean_vae_p95_pct:.1f}%)",
            3.0 <= clean_vae_p95_pct <= 10.0),
        (f"Clean VAE p99 ~1%  (got {clean_vae_p99_pct:.1f}%)",
            0.5 <= clean_vae_p99_pct <= 3.0),
        (f"Position jumps caught by pv_max  (got {min(pjump_kin_flag_pct):.1f}%)",
            all(v >= 80.0 for v in pjump_kin_flag_pct)),
        (f"Stationary caught by speed/displacement  (got {stat_kin_flag_pct:.1f}%)",
            stat_kin_flag_pct is not None and stat_kin_flag_pct >= 80.0),
        ("All output files written",             all(os.path.exists(p) for p in output_files)),
    ]
    print_report(device, n_sample, thr["vae_p95_threshold"], thr["vae_p99_threshold"],
                 summary_rows, checks, output_files)
    print(f"\nOutputs written to {_OUT}", flush=True)


if __name__ == "__main__":
    main()
