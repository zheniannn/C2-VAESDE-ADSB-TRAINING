"""Output helpers for the calibration stage."""

import os
import numpy as np
import pandas as pd


def save_thresholds(thresholds: dict, thr_rows: list[dict],
                    quantiles_df: pd.DataFrame, output_dir: str) -> tuple[str, str]:
    """Save calibrated thresholds and clean kinematic quantiles; return (thr_path, q_path)."""
    thr_path = os.path.join(output_dir, "calibrated_thresholds.csv")
    pd.DataFrame(thr_rows).to_csv(thr_path, index=False)
    q_path = os.path.join(output_dir, "clean_kinematic_quantiles.csv")
    quantiles_df.to_csv(q_path, index=False)
    return thr_path, q_path


def save_outputs(summary_rows: list[dict],
                 per_seq_chunks: list[pd.DataFrame],
                 example_list: list[dict],
                 output_dir: str) -> tuple[str, str, str]:
    """Save summary CSV, per-sequence CSV, and examples NPZ; return (sum, pseq, npz) paths."""
    sum_path = os.path.join(output_dir, "calibrated_motion_prior_summary.csv")
    pd.DataFrame(summary_rows).to_csv(sum_path, index=False)

    pseq_path = os.path.join(output_dir, "calibrated_motion_prior_per_sequence_scores.csv")
    pd.concat(per_seq_chunks, ignore_index=True).to_csv(pseq_path, index=False)

    npz_path = os.path.join(output_dir, "calibrated_motion_prior_examples.npz")
    np.savez(
        npz_path,
        test_types               = np.array([d["test_type"] for d in example_list]),
        sequence_indices         = np.stack([d["sequence_indices"] for d in example_list]),
        clean_physical           = np.stack([d["clean_physical"] for d in example_list]),
        corrupted_physical       = np.stack([d["corrupted_physical"] for d in example_list]),
        reconstructed_normalised = np.stack([d["reconstructed_normalised"] for d in example_list]),
    )
    return sum_path, pseq_path, npz_path


def build_summary_row(name: str, n: int, recon_mse: np.ndarray,
                      kin: dict, flags: dict) -> dict:
    """Build a per-test-type summary dict for the calibrated stress-test run."""
    pct = lambda a: float(100 * a.mean())
    p   = lambda a, q: float(np.percentile(a, q))
    return {
        "test_type": name, "num_sequences": n,
        "percent_vae_p95":              pct(flags["vae_flag_p95"]),
        "percent_vae_p99":              pct(flags["vae_flag_p99"]),
        "percent_kinematic_flagged":    pct(flags["kinematic_flag"]),
        "percent_combined_p95":         pct(flags["combined_flag_p95"]),
        "percent_combined_p99":         pct(flags["combined_flag_p99"]),
        "percent_flag_speed":           pct(flags["flag_speed"]),
        "percent_flag_accel":           pct(flags["flag_accel"]),
        "percent_flag_pv_max":          pct(flags["flag_pv_max"]),
        "percent_flag_pv_mean":         pct(flags["flag_pv_mean"]),
        "percent_flag_turn_max":        pct(flags["flag_turn_max"]),
        "percent_flag_turn_mean":       pct(flags["flag_turn_mean"]),
        "percent_flag_too_slow":        pct(flags["flag_too_slow"]),
        "percent_flag_low_displacement": pct(flags["flag_low_displacement"]),
        "recon_mse_mean": float(recon_mse.mean()),
        "recon_mse_p50":  p(recon_mse, 50), "recon_mse_p95": p(recon_mse, 95),
        "recon_mse_p99":  p(recon_mse, 99),
        "max_speed_p95":  p(kin["max_speed_mps"], 95),
        "max_speed_p99":  p(kin["max_speed_mps"], 99),
        "max_accel_p95":  p(kin["max_accel_mps2"], 95),
        "max_accel_p99":  p(kin["max_accel_mps2"], 99),
        "max_pv_error_p95": p(kin["max_pv_error_m"], 95),
        "max_pv_error_p99": p(kin["max_pv_error_m"], 99),
        "max_turn_rate_p95": p(kin["max_turn_rate_degps"], 95),
        "max_turn_rate_p99": p(kin["max_turn_rate_degps"], 99),
        "total_displacement_p5":  p(kin["total_displacement_m"],  5),
        "total_displacement_p50": p(kin["total_displacement_m"], 50),
        "total_displacement_p95": p(kin["total_displacement_m"], 95),
    }


def print_report(device, n_sample: int, thresholds: dict,
                 summary_rows: list[dict], checks: list[tuple],
                 output_files: list[str]) -> None:
    """Print final calibration report to stdout."""
    print()
    print("=" * 78)
    print("CALIBRATED MOTION PRIOR REPORT")
    print("=" * 78)
    print(f"  Device                    : {device}")
    print(f"  Test sequences sampled    : {n_sample:,}")
    print()
    print("Calibrated thresholds:")
    for k, v in thresholds.items():
        print(f"  {k:<42} {v:>10.3f}")
    print()
    hdr = (f"  {'Test type':<28}  {'VAE p95':>8}  {'VAE p99':>8}  "
           f"{'Kinem':>7}  {'Comb p95':>9}  {'Comb p99':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in summary_rows:
        print(f"  {r['test_type']:<28}  "
              f"{r['percent_vae_p95']:>7.1f}%  "
              f"{r['percent_vae_p99']:>7.1f}%  "
              f"{r['percent_kinematic_flagged']:>6.1f}%  "
              f"{r['percent_combined_p95']:>8.1f}%  "
              f"{r['percent_combined_p99']:>8.1f}%")
    print()
    print("Kinematic flag breakdown (calibrated):")
    fcols = ["flag_speed", "flag_accel", "flag_pv_max", "flag_pv_mean",
             "flag_turn_max", "flag_turn_mean", "flag_too_slow", "flag_low_displacement"]
    hdr2 = (f"  {'Test type':<28}  " +
            "  ".join(f"{c.replace('flag_',''):>10}" for c in fcols))
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for r in summary_rows:
        vals = "  ".join(f"{r.get('percent_' + c, 0):>9.1f}%" for c in fcols)
        print(f"  {r['test_type']:<28}  {vals}")
    print()
    print("Interpretation:")
    _interpret(summary_rows)
    print()
    print("Sanity checks:")
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    print()
    print("Files written:")
    for p in output_files:
        print(f"  {p}  ({os.path.getsize(p) / 1e6:.1f} MB)")


def _interpret(rows: list[dict]) -> None:
    clean_v = rows[0]["percent_vae_p95"]
    for r in rows[1:]:
        name   = r["test_type"]
        v95    = r["percent_vae_p95"]
        k_pct  = r["percent_kinematic_flagged"]
        c95    = r["percent_combined_p95"]
        k_gain = c95 - v95
        dominant_flags = [
            col.replace("percent_flag_", "")
            for col in ["percent_flag_speed", "percent_flag_accel",
                        "percent_flag_pv_max", "percent_flag_pv_mean",
                        "percent_flag_turn_max", "percent_flag_turn_mean",
                        "percent_flag_too_slow", "percent_flag_low_displacement"]
            if r.get(col, 0) >= 5.0
        ]
        dom = ", ".join(dominant_flags) if dominant_flags else "none"
        verdict = ("combined >> VAE"       if k_gain >= 20
                   else "VAE dominant"     if v95 >= 80
                   else "kinematic dominant" if k_pct >= 80
                   else "partial both")
        print(f"  {name:<28}  VAE={v95:.0f}%  Kin={k_pct:.0f}%  "
              f"Comb={c95:.0f}%  [{dom}]  → {verdict}")
