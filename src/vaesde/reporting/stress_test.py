"""Output helpers for the stress-test stage."""

import os
import numpy as np
import pandas as pd


def score_mse(mse: np.ndarray, p95_thr: float, p99_thr: float) -> dict:
    """Compute MSE distribution stats and threshold exceedance rates."""
    return {
        "num_sequences":                len(mse),
        "mean_mse":                     float(np.mean(mse)),
        "median_mse":                   float(np.median(mse)),
        "p90_mse":                      float(np.percentile(mse, 90)),
        "p95_mse":                      float(np.percentile(mse, 95)),
        "p99_mse":                      float(np.percentile(mse, 99)),
        "max_mse":                      float(np.max(mse)),
        "percent_above_p95_threshold":  float(100 * np.mean(mse > p95_thr)),
        "percent_above_p99_threshold":  float(100 * np.mean(mse > p99_thr)),
    }


def save_outputs(results: dict[str, dict],
                 all_mse: dict[str, np.ndarray],
                 meta: pd.DataFrame,
                 example_list: list[dict],
                 p95_threshold: float,
                 p99_threshold: float,
                 output_dir: str) -> list[str]:
    """Save stress-test summary CSV, per-sequence CSV, and examples NPZ.

    Returns list of written file paths.
    """
    summary_df = pd.DataFrame(list(results.values()))
    summary_path = os.path.join(output_dir, "stress_test_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    chunks = []
    for test_name, mse in all_mse.items():
        df = meta[["sequence_id", "segment_id", "start_time", "end_time"]].copy()
        df["test_type"]           = test_name
        df["recon_mse"]           = mse
        df["above_p95_threshold"] = mse > p95_threshold
        df["above_p99_threshold"] = mse > p99_threshold
        chunks.append(df)
    per_seq_path = os.path.join(output_dir, "stress_test_per_sequence_errors.csv")
    pd.concat(chunks, ignore_index=True).to_csv(per_seq_path, index=False)

    npz_path = os.path.join(output_dir, "stress_test_examples.npz")
    np.savez(
        npz_path,
        test_types               = np.array([d["test_type"] for d in example_list]),
        sequence_indices         = np.stack([d["sequence_indices"] for d in example_list]),
        original_clean_physical  = np.stack([d["original_clean_physical"] for d in example_list]),
        corrupted_physical       = np.stack([d["corrupted_physical"] for d in example_list]),
        reconstructed_normalised = np.stack([d["reconstructed_normalised"] for d in example_list]),
    )
    return [summary_path, per_seq_path, npz_path]


def print_report(device, n_sample: int, p95_threshold: float, p99_threshold: float,
                 results: dict[str, dict], checks: list[tuple],
                 output_files: list[str]) -> None:
    """Print final stress-test report to stdout."""
    print()
    print("=" * 72)
    print("STRESS TEST REPORT")
    print("=" * 72)
    print(f"  Device                  : {device}")
    print(f"  Test sequences sampled  : {n_sample:,}")
    print(f"  p95 threshold           : {p95_threshold}")
    print(f"  p99 threshold           : {p99_threshold}")
    print()
    print(f"  {'Test type':<28}  {'mean MSE':>9}  {'p50':>8}  "
          f"{'p95':>8}  {'>p95 %':>7}  {'>p99 %':>7}")
    print("  " + "-" * 72)
    for r in results.values():
        print(f"  {r['test_type']:<28}  {r['mean_mse']:>9.5f}  "
              f"{r['median_mse']:>8.5f}  {r['p95_mse']:>8.5f}  "
              f"{r['percent_above_p95_threshold']:>6.1f}%  "
              f"{r['percent_above_p99_threshold']:>6.1f}%")
    print()
    print("Interpretation:")
    _interpret(results)
    print()
    print("Sanity checks:")
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    print()
    print("Files written:")
    for p in output_files:
        size = os.path.getsize(p) / 1e6
        print(f"  {p}  ({size:.1f} MB)")


def _interpret(results: dict[str, dict]) -> None:
    clean_mean = results["clean"]["mean_mse"]
    for name, r in results.items():
        if name == "clean":
            continue
        ratio   = r["mean_mse"] / clean_mean if clean_mean > 0 else 0.0
        pct_p95 = r["percent_above_p95_threshold"]
        if pct_p95 >= 80:
            verdict = "caught strongly    (>80% above p95)"
        elif pct_p95 >= 40:
            verdict = "caught moderately  (40–80% above p95)"
        elif pct_p95 >= 10:
            verdict = "partially detected (10–40% above p95)"
        else:
            verdict = "not well detected  (<10% above p95)"
        print(f"  {name:<28}  {ratio:>4.1f}x clean  "
              f"{pct_p95:>5.1f}% >p95  →  {verdict}")
