"""Output helpers for the motion-prior scorer stage."""

import os
import numpy as np
import pandas as pd


def save_outputs(summary_rows: list[dict],
                 per_seq_chunks: list[pd.DataFrame],
                 example_list: list[dict],
                 output_dir: str) -> list[str]:
    """Save summary CSV, per-sequence CSV, and examples NPZ; return file paths."""
    summary_path = os.path.join(output_dir, "motion_prior_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    per_seq_path = os.path.join(output_dir, "motion_prior_per_sequence_scores.csv")
    pd.concat(per_seq_chunks, ignore_index=True).to_csv(per_seq_path, index=False)

    npz_path = os.path.join(output_dir, "motion_prior_examples.npz")
    np.savez(
        npz_path,
        test_types               = np.array([d["test_type"] for d in example_list]),
        sequence_indices         = np.stack([d["sequence_indices"] for d in example_list]),
        clean_physical           = np.stack([d["clean_physical"] for d in example_list]),
        corrupted_physical       = np.stack([d["corrupted_physical"] for d in example_list]),
        reconstructed_normalised = np.stack([d["reconstructed_normalised"] for d in example_list]),
    )
    return [summary_path, per_seq_path, npz_path]


def build_summary_row(test_type: str, recon_mse: np.ndarray,
                      kin: dict, flags: dict) -> dict:
    """Build a per-test-type summary dict for the motion-prior scorer run."""
    pct = lambda arr: float(100 * arr.mean())
    p   = lambda arr, q: float(np.percentile(arr, q))
    return {
        "test_type": test_type,
        "percent_vae_above_p95":          pct(flags["vae_flag_p95"]),
        "percent_vae_above_p99":          pct(flags["vae_flag_p99"]),
        "percent_kinematic_flagged":       pct(flags["kinematic_flag"]),
        "percent_flag_speed":              pct(flags["flag_speed"]),
        "percent_flag_accel":              pct(flags["flag_accel"]),
        "percent_flag_pv_max":             pct(flags["flag_pv_max"]),
        "percent_flag_pv_mean":            pct(flags["flag_pv_mean"]),
        "percent_flag_too_slow":           pct(flags["flag_too_slow"]),
        "percent_flag_low_displacement":   pct(flags["flag_low_displacement"]),
        "percent_combined_p95_flagged":    pct(flags["combined_flag_p95"]),
        "percent_combined_p99_flagged":    pct(flags["combined_flag_p99"]),
        "recon_mse_mean": float(recon_mse.mean()),
        "recon_mse_p50":  p(recon_mse, 50), "recon_mse_p95": p(recon_mse, 95),
        "recon_mse_p99":  p(recon_mse, 99),
        "max_speed_mean": float(kin["max_speed_mps"].mean()),
        "max_speed_p95":  p(kin["max_speed_mps"], 95),
        "max_speed_p99":  p(kin["max_speed_mps"], 99),
        "max_speed_max":  float(kin["max_speed_mps"].max()),
        "max_accel_mean": float(kin["max_accel_mps2"].mean()),
        "max_accel_p95":  p(kin["max_accel_mps2"], 95),
        "max_accel_p99":  p(kin["max_accel_mps2"], 99),
        "max_accel_max":  float(kin["max_accel_mps2"].max()),
        "max_pv_error_mean": float(kin["max_pv_error_m"].mean()),
        "max_pv_error_p95":  p(kin["max_pv_error_m"], 95),
        "max_pv_error_p99":  p(kin["max_pv_error_m"], 99),
        "max_pv_error_max":  float(kin["max_pv_error_m"].max()),
        "mean_pv_error_mean": float(kin["mean_pv_error_m"].mean()),
        "mean_pv_error_p95":  p(kin["mean_pv_error_m"], 95),
        "mean_pv_error_p99":  p(kin["mean_pv_error_m"], 99),
        "mean_pv_error_max":  float(kin["mean_pv_error_m"].max()),
        "displacement_mean": float(kin["total_displacement_m"].mean()),
        "displacement_p5":   p(kin["total_displacement_m"],  5),
        "displacement_p50":  p(kin["total_displacement_m"], 50),
        "displacement_p95":  p(kin["total_displacement_m"], 95),
    }


def print_report(device, n_sample: int, vae_p95: float, vae_p99: float,
                 summary_rows: list[dict], checks: list[tuple],
                 output_files: list[str]) -> None:
    """Print final motion-prior scorer report to stdout."""
    print()
    print("=" * 78)
    print("MOTION PRIOR SCORER REPORT")
    print("=" * 78)
    print(f"  Device                    : {device}")
    print(f"  Test sequences sampled    : {n_sample:,}")
    print(f"  VAE p95 threshold         : {vae_p95}")
    print(f"  VAE p99 threshold         : {vae_p99}")
    print()
    hdr = (f"  {'Test type':<28}  {'VAE p95':>8}  {'VAE p99':>8}  "
           f"{'Kinem':>7}  {'Comb p95':>9}  {'Comb p99':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in summary_rows:
        print(f"  {r['test_type']:<28}  "
              f"{r['percent_vae_above_p95']:>7.1f}%  "
              f"{r['percent_vae_above_p99']:>7.1f}%  "
              f"{r['percent_kinematic_flagged']:>6.1f}%  "
              f"{r['percent_combined_p95_flagged']:>8.1f}%  "
              f"{r['percent_combined_p99_flagged']:>8.1f}%")
    print()
    print("Kinematic flag breakdown:")
    flag_cols = ["flag_speed", "flag_accel", "flag_pv_max",
                 "flag_pv_mean", "flag_too_slow", "flag_low_displacement"]
    hdr2 = (f"  {'Test type':<28}  " +
            "  ".join(f"{c.replace('flag_',''):>12}" for c in flag_cols))
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for r in summary_rows:
        vals = "  ".join(f"{r.get('percent_' + c, 0):>11.1f}%" for c in flag_cols)
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
    clean_vae = rows[0]["percent_vae_above_p95"]
    for r in rows[1:]:
        name   = r["test_type"]
        v_gain = r["percent_vae_above_p95"] - clean_vae
        k_pct  = r["percent_kinematic_flagged"]
        c_pct  = r["percent_combined_p95_flagged"]
        gain   = c_pct - r["percent_vae_above_p95"]
        flags_hit = [
            col.replace("percent_flag_", "")
            for col in ["percent_flag_speed", "percent_flag_accel",
                        "percent_flag_pv_max", "percent_flag_pv_mean",
                        "percent_flag_too_slow", "percent_flag_low_displacement"]
            if r.get(col, 0) >= 10.0
        ]
        dom = ", ".join(flags_hit) if flags_hit else "none"
        verdict = ("combined adds value"      if gain >= 5.0
                   else "VAE alone sufficient" if r["percent_vae_above_p95"] >= 80.0
                   else "kinematic fills gap"  if k_pct >= 80.0
                   else "partial detection only")
        print(f"  {name:<28}  VAE+{v_gain:+.0f}%  Kin={k_pct:.0f}%  "
              f"Comb={c_pct:.0f}%  flags=[{dom}]  → {verdict}")
