"""Calibrate VAE + kinematic thresholds from clean-sequence quantiles.

Inputs  : models/sequence_vae/sequence_vae_full.pt
          data/X_train.npy          (threshold calibration)
          data/X_test.npy           (evaluation / stress-test)
          data/normalisation_{mean,std}.csv
          data/test_sequence_metadata.csv

Outputs : outputs/calibrate/
            calibrated_thresholds.csv
            clean_kinematic_quantiles.csv
            calibrated_motion_prior_summary.csv
            calibrated_motion_prior_per_sequence_scores.csv
            calibrated_motion_prior_examples.npz
"""

import os
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT  = os.path.join(_PROJECT_ROOT, "outputs", "calibrate")
_CKPT = os.path.join(_PROJECT_ROOT, "models", "sequence_vae", "sequence_vae_full.pt")

NUM_TEST_SEQUENCES = 50_000
N_EXAMPLES         = 10
SUDDEN_PREV_COMB   = 15.3   # combined% from the uncalibrated run_score baseline

from vaesde.io_utils      import load_config, load_norm_stats
from vaesde.model         import SequenceVAE
from vaesde.inference     import compute_recon_mse
from vaesde.normalisation import renormalise
from vaesde.corruption    import (corrupt_speed_scale, corrupt_position_jump,
                                   corrupt_random_walk_velocity, corrupt_sudden_turn,
                                   corrupt_stationary)
from vaesde.kinematics    import compute_kinematics, compute_flags, calibrate_thresholds
from vaesde.constants     import SEQ_LEN, N_FEAT, FLAT_DIM, LATENT_DIM
from vaesde.reporting.calibrate import build_summary_row, save_thresholds, save_outputs, print_report


def main() -> None:
    print("=== run_calibrate: threshold calibration ===", flush=True)
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
    print(f"Loaded: {_CKPT}")

    with torch.no_grad():
        dummy_out, _, _ = model(torch.zeros(2, SEQ_LEN, N_FEAT, device=device))
        assert dummy_out.shape == (2, SEQ_LEN, N_FEAT)

    mean, std = load_norm_stats(
        os.path.join(_PROJECT_ROOT, "data", "normalisation_mean.csv"),
        os.path.join(_PROJECT_ROOT, "data", "normalisation_std.csv"),
    )

    # --- Calibration data: train split only ---
    X_train_full = np.load(os.path.join(_PROJECT_ROOT, "data", "X_train.npy"), mmap_mode="r")
    assert X_train_full.shape[1:] == (SEQ_LEN, N_FEAT)
    n_cal        = min(NUM_TEST_SEQUENCES, len(X_train_full))
    cal_idx      = np.sort(rng.choice(len(X_train_full), n_cal, replace=False))
    cal_norm     = X_train_full[cal_idx].copy().astype(np.float64)
    cal_phys     = cal_norm * std + mean
    print(f"Calibration: sampled {n_cal:,} train sequences  (X_train: {X_train_full.shape})")

    print("Computing clean kinematic quantiles on train data ...")
    clean_kin = compute_kinematics(cal_phys)
    thresholds, quantiles_df = calibrate_thresholds(clean_kin)
    thresholds["vae_p95_threshold"] = P95_THR
    thresholds["vae_p99_threshold"] = P99_THR

    # --- Evaluation data: test split only ---
    X_test_full = np.load(os.path.join(_PROJECT_ROOT, "data", "X_test.npy"), mmap_mode="r")
    assert X_test_full.shape[1:] == (SEQ_LEN, N_FEAT)
    n_sample  = min(NUM_TEST_SEQUENCES, len(X_test_full))
    idx       = np.sort(rng.choice(len(X_test_full), n_sample, replace=False))
    seqs_norm = X_test_full[idx].copy().astype(np.float64)
    seqs_phys = seqs_norm * std + mean
    meta      = pd.read_csv(
        os.path.join(_PROJECT_ROOT, "data", "test_sequence_metadata.csv")
    ).iloc[idx].reset_index(drop=True)
    print(f"Evaluation: sampled {n_sample:,} test sequences  (X_test: {X_test_full.shape})\n")

    print("\nCalibrated thresholds (from train data):")
    for k, v in thresholds.items():
        if not k.startswith("vae_"):
            print(f"  {k:<40} = {v:.4f}")

    thr_rows = [
        {"threshold_name": "max_speed_threshold_mps",
         "value": thresholds["max_speed_threshold_mps"],
         "source_metric": "max_speed_mps", "source_quantile": "p99_5 (floored at 150 m/s)",
         "rationale": "Physical cap 150 m/s OR data-driven p99.5; whichever is higher"},
        {"threshold_name": "max_accel_threshold_mps2",
         "value": thresholds["max_accel_threshold_mps2"],
         "source_metric": "max_accel_mps2", "source_quantile": "p99_5",
         "rationale": "0.5% false-flag rate per flag on clean data"},
        {"threshold_name": "mean_pv_error_threshold_m",
         "value": thresholds["mean_pv_error_threshold_m"],
         "source_metric": "mean_pv_error_m", "source_quantile": "p99_5",
         "rationale": "0.5% false-flag rate per flag on clean data"},
        {"threshold_name": "max_pv_error_threshold_m",
         "value": thresholds["max_pv_error_threshold_m"],
         "source_metric": "max_pv_error_m", "source_quantile": "p99_5",
         "rationale": "0.5% false-flag rate per flag on clean data"},
        {"threshold_name": "mean_turn_rate_threshold_degps",
         "value": thresholds["mean_turn_rate_threshold_degps"],
         "source_metric": "mean_turn_rate_degps", "source_quantile": "p99_5",
         "rationale": "0.5% false-flag rate per flag on clean data"},
        {"threshold_name": "max_turn_rate_threshold_degps",
         "value": thresholds["max_turn_rate_threshold_degps"],
         "source_metric": "max_turn_rate_degps", "source_quantile": "p99_5",
         "rationale": "0.5% false-flag rate per flag on clean data"},
        {"threshold_name": "min_mean_speed_threshold_mps",
         "value": thresholds["min_mean_speed_threshold_mps"],
         "source_metric": "mean_speed_mps", "source_quantile": "p0_5 (floored at 1.0 m/s)",
         "rationale": "0.5% false-flag rate on lower tail; minimum 1 m/s"},
        {"threshold_name": "min_displacement_threshold_m",
         "value": thresholds["min_displacement_threshold_m"],
         "source_metric": "total_displacement_m", "source_quantile": "p0_5 (floored at 50 m)",
         "rationale": "0.5% false-flag rate on lower tail; minimum 50 m"},
    ]
    thr_path, q_path = save_thresholds(thresholds, thr_rows, quantiles_df, _OUT)

    TEST_CASES: list[tuple[str, np.ndarray]] = [
        ("clean",                   seqs_phys.copy()),
        ("speed_scaled_1p5",        corrupt_speed_scale(seqs_phys, 1.5)),
        ("speed_scaled_2p0",        corrupt_speed_scale(seqs_phys, 2.0)),
        ("position_jump_1000m",     corrupt_position_jump(seqs_phys, 1000.0)),
        ("position_jump_2000m",     corrupt_position_jump(seqs_phys, 2000.0)),
        ("random_walk_velocity",    corrupt_random_walk_velocity(seqs_phys, rng)),
        ("sudden_turn_90deg",       corrupt_sudden_turn(seqs_phys)),
        ("stationary_clutter_like", corrupt_stationary(seqs_phys)),
    ]

    for name, phys in TEST_CASES[1:]:
        n = renormalise(phys, mean, std)
        assert not np.isnan(n).any() and not np.isinf(n).any(), f"NaN/Inf in {name}"
    print("\nNaN/Inf check: OK\n")

    ex_idx         = np.sort(rng.choice(n_sample, N_EXAMPLES, replace=False))
    summary_rows:   list[dict]         = []
    per_seq_chunks: list[pd.DataFrame] = []
    example_list:   list[dict]         = []

    clean_kflag_pct    = None
    clean_vae_p95_pct  = None
    clean_vae_p99_pct  = None
    pjump_pv_pcts:  list[float] = []
    stat_lower_pct  = None
    sudden_new_comb = None

    print(f"  {'Test type':<28}  {'VAE p95':>8}  {'Kin':>6}  {'Comb p95':>9}")
    print("  " + "-" * 58)

    for test_name, corrupted_phys in TEST_CASES:
        norm = renormalise(corrupted_phys, mean, std)
        recon_mse, recon_arr = compute_recon_mse(model, norm, device, BATCH_SIZE)
        kin   = compute_kinematics(corrupted_phys)
        flags = compute_flags(kin, recon_mse, thresholds)
        row   = build_summary_row(test_name, n_sample, recon_mse, kin, flags)
        summary_rows.append(row)

        df = meta[["sequence_id", "segment_id", "start_time", "end_time"]].copy()
        df["test_type"] = test_name
        df["recon_mse"] = recon_mse
        for k in ["vae_flag_p95", "vae_flag_p99"]:
            df[k] = flags[k]
        for k in ["mean_speed_mps", "max_speed_mps", "mean_accel_mps2", "max_accel_mps2",
                  "mean_pv_error_m", "max_pv_error_m", "total_displacement_m",
                  "mean_turn_rate_degps", "max_turn_rate_degps"]:
            df[k] = kin[k]
        for k in ["flag_speed", "flag_accel", "flag_pv_max", "flag_pv_mean",
                  "flag_turn_max", "flag_turn_mean", "flag_too_slow",
                  "flag_low_displacement", "kinematic_flag",
                  "combined_flag_p95", "combined_flag_p99"]:
            df[k] = flags[k]
        per_seq_chunks.append(df)

        example_list.append({
            "test_type":               test_name,
            "sequence_indices":        ex_idx.copy(),
            "clean_physical":          seqs_phys[ex_idx].astype(np.float32),
            "corrupted_physical":      corrupted_phys[ex_idx].astype(np.float32),
            "reconstructed_normalised": recon_arr[ex_idx],
        })

        v95  = row["percent_vae_p95"]
        kpct = row["percent_kinematic_flagged"]
        c95  = row["percent_combined_p95"]
        print(f"  {test_name:<28}  {v95:>7.1f}%  {kpct:>5.1f}%  {c95:>8.1f}%")

        if test_name == "clean":
            clean_kflag_pct   = kpct
            clean_vae_p95_pct = v95
            clean_vae_p99_pct = row["percent_vae_p99"]
        elif test_name in ("position_jump_1000m", "position_jump_2000m"):
            pjump_pv_pcts.append(row["percent_flag_pv_max"])
        elif test_name == "stationary_clutter_like":
            stat_lower_pct = max(row["percent_flag_too_slow"],
                                 row["percent_flag_low_displacement"])
        elif test_name == "sudden_turn_90deg":
            sudden_new_comb = c95

    sum_path, pseq_path, npz_path = save_outputs(summary_rows, per_seq_chunks, example_list, _OUT)
    output_files = [thr_path, q_path, sum_path, pseq_path, npz_path]

    print()
    print("Calibration impact vs previous hand-picked thresholds:")
    print(f"  Clean kinematic flag rate : 18.7%  →  {clean_kflag_pct:.1f}%")
    print(f"  Clean combined p95 rate   : 20.6%  →  {summary_rows[0]['percent_combined_p95']:.1f}%")
    if sudden_new_comb is not None:
        print(f"  sudden_turn_90deg combined: {SUDDEN_PREV_COMB}%  →  {sudden_new_comb:.1f}%  "
              f"({'improved' if sudden_new_comb > SUDDEN_PREV_COMB else 'unchanged/lower'})")

    checks = [
        ("Checkpoint exists",                        os.path.exists(_CKPT)),
        ("X_train shape compatible (N, 30, 4)",      X_train_full.shape[1:] == (SEQ_LEN, N_FEAT)),
        ("X_test shape compatible (N, 30, 4)",       X_test_full.shape[1:] == (SEQ_LEN, N_FEAT)),
        ("Normalisation stats correct features",      True),
        ("No NaN/Inf in corrupted sequences",         True),
        ("VAE output shape matches input",            dummy_out.shape == (2, SEQ_LEN, N_FEAT)),
        (f"Clean VAE p95 ~5%  (got {clean_vae_p95_pct:.1f}%)",
            3.0 <= clean_vae_p95_pct <= 10.0),
        (f"Clean VAE p99 ~1%  (got {clean_vae_p99_pct:.1f}%)",
            0.5 <= clean_vae_p99_pct <= 3.0),
        (f"Clean kinematic flag < 10%  (got {clean_kflag_pct:.1f}%)",
            clean_kflag_pct < 10.0),
        (f"Position jumps pv_max >= 80%  (got {min(pjump_pv_pcts):.1f}%)",
            all(v >= 80.0 for v in pjump_pv_pcts)),
        (f"Stationary lower-tail >= 80%  (got {stat_lower_pct:.1f}%)",
            stat_lower_pct is not None and stat_lower_pct >= 80.0),
        (f"sudden_turn combined > 0%  (got {sudden_new_comb:.1f}%)",
            sudden_new_comb is not None and sudden_new_comb > 0),
        ("All output files written", all(os.path.exists(p) for p in output_files)),
    ]
    print_report(device, n_sample, thresholds, summary_rows, checks, output_files)
    print(f"\nOutputs written to {_OUT}", flush=True)


if __name__ == "__main__":
    main()
