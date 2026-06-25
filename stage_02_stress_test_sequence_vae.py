"""
stage_02_stress_test_sequence_vae.py

Stress-tests the trained sequence VAE by comparing reconstruction error on:
  - clean test sequences (baseline)
  - eight categories of artificially corrupted / physically implausible trajectories

Does NOT retrain the model.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_TEST_SEQUENCES = 50_000
SEED               = 42
P95_THRESHOLD      = 0.03948
P99_THRESHOLD      = 0.10405
DT                 = 10.0          # seconds between timesteps
BATCH_SIZE         = 1024
N_EXAMPLES         = 10            # example sequences saved per test type

SEQ_LEN    = 30
N_FEAT     = 4
FLAT_DIM   = SEQ_LEN * N_FEAT      # 120
LATENT_DIM = 16

# Feature column indices: [E_m, N_m, vE_mps, vN_mps]
IDX_E  = 0
IDX_N  = 1
IDX_VE = 2
IDX_VN = 3
FEATURES = ["E_m", "N_m", "vE_mps", "vN_mps"]

OUT_DIR   = Path("models/sequence_vae/stress_tests")
CKPT_PATH = Path("models/sequence_vae/sequence_vae_full.pt")


# ---------------------------------------------------------------------------
# Model  (identical architecture to stage_01_train_sequence_vae.py)
# ---------------------------------------------------------------------------
class SequenceVAE(nn.Module):
    def __init__(self, flat_dim: int = FLAT_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),      nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder   = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),        nn.ReLU(),
            nn.Linear(256, flat_dim),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x: torch.Tensor):
        b           = x.size(0)
        x_flat      = x.view(b, FLAT_DIM)
        mu, logvar  = self.encode(x_flat)
        z           = self.reparameterise(mu, logvar)
        recon_flat  = self.decoder(z)
        return recon_flat.view(b, SEQ_LEN, N_FEAT), mu, logvar


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def denormalise(seqs_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return seqs_norm * std + mean           # broadcast over (N, 30, 4)

def renormalise(seqs_phys: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (seqs_phys - mean) / std


# ---------------------------------------------------------------------------
# Corruption helpers  (all operate on physical-unit arrays, return physical)
# ---------------------------------------------------------------------------
def _integrate_velocities(s: np.ndarray) -> np.ndarray:
    """
    Rebuild E_m and N_m from vE_mps / vN_mps starting from the original first point.
    E_m[t] = E_m[0] + sum_{k=0}^{t-1} vE[k] * DT   (for t = 1..29)
    Modifies s in-place and returns it.
    """
    E0 = s[:, 0:1, IDX_E]                           # (N, 1)
    N0 = s[:, 0:1, IDX_N]
    s[:, 1:, IDX_E] = E0 + np.cumsum(s[:, :-1, IDX_VE], axis=1) * DT
    s[:, 1:, IDX_N] = N0 + np.cumsum(s[:, :-1, IDX_VN], axis=1) * DT
    return s


def corrupt_speed_scale(seqs_phys: np.ndarray, factor: float) -> np.ndarray:
    """Scale vE and vN by factor, then reintegrate positions."""
    s = seqs_phys.copy()
    s[:, :, IDX_VE] *= factor
    s[:, :, IDX_VN] *= factor
    return _integrate_velocities(s)


def corrupt_position_jump(seqs_phys: np.ndarray, jump_m: float) -> np.ndarray:
    """Add a discontinuous +jump_m to E_m from timestep 15 onward; velocities unchanged."""
    s = seqs_phys.copy()
    s[:, 15:, IDX_E] += jump_m
    return s


def corrupt_random_walk_velocity(seqs_phys: np.ndarray,
                                 rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise (std=20 m/s) to vE and vN, then reintegrate positions."""
    s   = seqs_phys.copy()
    N   = s.shape[0]
    s[:, :, IDX_VE] += rng.normal(0.0, 20.0, (N, SEQ_LEN))
    s[:, :, IDX_VN] += rng.normal(0.0, 20.0, (N, SEQ_LEN))
    return _integrate_velocities(s)


def corrupt_sudden_turn(seqs_phys: np.ndarray) -> np.ndarray:
    """Rotate velocity vector 90° from timestep 15 onward, then reintegrate all."""
    s        = seqs_phys.copy()
    vE_orig  = s[:, :, IDX_VE].copy()
    vN_orig  = s[:, :, IDX_VN].copy()
    s[:, 15:, IDX_VE] = -vN_orig[:, 15:]
    s[:, 15:, IDX_VN] =  vE_orig[:, 15:]
    return _integrate_velocities(s)


def corrupt_stationary(seqs_phys: np.ndarray) -> np.ndarray:
    """Fix E_m and N_m to timestep-0 values; set all velocities to zero."""
    s = seqs_phys.copy()
    s[:, :, IDX_E]  = s[:, 0:1, IDX_E]
    s[:, :, IDX_N]  = s[:, 0:1, IDX_N]
    s[:, :, IDX_VE] = 0.0
    s[:, :, IDX_VN] = 0.0
    return s


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def compute_recon_mse(model: SequenceVAE,
                      seqs_norm: np.ndarray,
                      device: torch.device) -> np.ndarray:
    """Per-sequence MSE averaged over (30, 4), shape (N,)."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(seqs_norm), BATCH_SIZE):
            b_np  = seqs_norm[start:start + BATCH_SIZE].astype(np.float32)
            b     = torch.from_numpy(b_np).to(device)
            recon, _, _ = model(b)
            mse   = ((recon - b) ** 2).mean(dim=(1, 2))
            chunks.append(mse.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def get_reconstructions(model: SequenceVAE,
                        seqs_norm: np.ndarray,
                        device: torch.device) -> np.ndarray:
    """Return reconstructed sequences (N, 30, 4) as float32 numpy."""
    model.eval()
    with torch.no_grad():
        t     = torch.from_numpy(seqs_norm.astype(np.float32)).to(device)
        recon, _, _ = model(t)
    return recon.cpu().numpy()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(mse: np.ndarray) -> dict:
    return {
        "num_sequences":                len(mse),
        "mean_mse":                     float(np.mean(mse)),
        "median_mse":                   float(np.median(mse)),
        "p90_mse":                      float(np.percentile(mse, 90)),
        "p95_mse":                      float(np.percentile(mse, 95)),
        "p99_mse":                      float(np.percentile(mse, 99)),
        "max_mse":                      float(np.max(mse)),
        "percent_above_p95_threshold":  float(100 * np.mean(mse > P95_THRESHOLD)),
        "percent_above_p99_threshold":  float(100 * np.mean(mse > P99_THRESHOLD)),
    }


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Sanity 1: checkpoint exists ---
    assert CKPT_PATH.exists(), f"Checkpoint not found: {CKPT_PATH}"
    model = SequenceVAE(FLAT_DIM, LATENT_DIM).to(device)
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model from {CKPT_PATH}")

    # --- Sanity 5: output shape ---
    with torch.no_grad():
        dummy     = torch.zeros(2, SEQ_LEN, N_FEAT, device=device)
        dummy_out, _, _ = model(dummy)
        assert dummy_out.shape == (2, SEQ_LEN, N_FEAT), \
            f"Output shape mismatch: {dummy_out.shape}"

    # --- Normalisation stats ---
    mean_df = pd.read_csv("data/normalisation_mean.csv", index_col=0)
    std_df  = pd.read_csv("data/normalisation_std.csv",  index_col=0)

    # --- Sanity 3: correct features ---
    for df, tag in [(mean_df, "mean"), (std_df, "std")]:
        missing = [f for f in FEATURES if f not in df.index]
        assert not missing, f"normalisation_{tag}.csv missing: {missing}"

    mean = mean_df.loc[FEATURES, "mean"].values.astype(np.float64)  # (4,)
    std  = std_df.loc[FEATURES,  "std"].values.astype(np.float64)

    # --- Load X_test (memory-mapped) ---
    X_test_full = np.load("data/X_test.npy", mmap_mode="r")
    print(f"X_test shape: {X_test_full.shape}")

    # --- Sanity 2: shape ---
    assert X_test_full.ndim == 3 and X_test_full.shape[1:] == (SEQ_LEN, N_FEAT)

    # --- Sample test sequences ---
    n_avail  = len(X_test_full)
    n_sample = min(NUM_TEST_SEQUENCES, n_avail)
    idx      = np.sort(rng.choice(n_avail, n_sample, replace=False))
    seqs_norm = X_test_full[idx].copy().astype(np.float64)   # (N, 30, 4) in RAM
    print(f"Sampled {n_sample:,} sequences (available: {n_avail:,})")

    meta_full = pd.read_csv("data/test_sequence_metadata.csv")
    meta      = meta_full.iloc[idx].reset_index(drop=True)

    # --- Physical sequences (denormalised) ---
    seqs_phys = denormalise(seqs_norm, mean, std)   # (N, 30, 4)

    # --- Build all test cases ---
    TEST_CASES: list[tuple[str, np.ndarray]] = [
        ("clean",
            seqs_norm.copy()),
        ("speed_scaled_1p5",
            renormalise(corrupt_speed_scale(seqs_phys.copy(), 1.5), mean, std)),
        ("speed_scaled_2p0",
            renormalise(corrupt_speed_scale(seqs_phys.copy(), 2.0), mean, std)),
        ("position_jump_1000m",
            renormalise(corrupt_position_jump(seqs_phys.copy(), 1000.0), mean, std)),
        ("position_jump_2000m",
            renormalise(corrupt_position_jump(seqs_phys.copy(), 2000.0), mean, std)),
        ("random_walk_velocity",
            renormalise(corrupt_random_walk_velocity(seqs_phys.copy(), rng), mean, std)),
        ("sudden_turn_90deg",
            renormalise(corrupt_sudden_turn(seqs_phys.copy()), mean, std)),
        ("stationary_clutter_like",
            renormalise(corrupt_stationary(seqs_phys.copy()), mean, std)),
    ]

    # --- Sanity 4: no NaN / Inf in corrupted sequences ---
    for name, arr in TEST_CASES[1:]:
        assert not np.isnan(arr).any(), f"NaN in {name}"
        assert not np.isinf(arr).any(), f"Inf in {name}"
    print("NaN/Inf check on all corrupted arrays: OK\n")

    # --- Choose fixed example indices (same across test types) ---
    ex_idx = np.sort(rng.choice(n_sample, N_EXAMPLES, replace=False))

    # --- Run inference ---
    results:      dict[str, dict]       = {}
    all_mse:      dict[str, np.ndarray] = {}
    example_list: list[dict]            = []

    print(f"  {'Test type':<28}  {'mean':>8}  {'p50':>8}  {'p95':>8}  "
          f"{'>p95%':>6}  {'>p99%':>6}")
    print("  " + "-" * 68)

    for test_name, arr_norm in TEST_CASES:
        mse = compute_recon_mse(model, arr_norm, device)
        results[test_name] = {"test_type": test_name, **score(mse)}
        all_mse[test_name] = mse

        r = results[test_name]
        print(f"  {test_name:<28}  {r['mean_mse']:>8.5f}  {r['median_mse']:>8.5f}  "
              f"{r['p95_mse']:>8.5f}  {r['percent_above_p95_threshold']:>5.1f}%  "
              f"{r['percent_above_p99_threshold']:>5.1f}%")

        # Collect examples (small batch, OK to load fully)
        ex_norm  = arr_norm[ex_idx].astype(np.float32)
        ex_recon = get_reconstructions(model, ex_norm, device)
        example_list.append({
            "test_type":                test_name,
            "sequence_indices":         ex_idx.copy(),
            "original_clean_physical":  seqs_phys[ex_idx].astype(np.float32),
            "corrupted_physical":       denormalise(arr_norm[ex_idx], mean, std).astype(np.float32),
            "reconstructed_normalised": ex_recon,
        })

    # --- Sanity 6: clean baseline ---
    clean_p95_pct = results["clean"]["percent_above_p95_threshold"]
    baseline_ok   = 3.0 <= clean_p95_pct <= 10.0

    # --- Save stress_test_summary.csv ---
    summary_df   = pd.DataFrame(list(results.values()))
    summary_path = OUT_DIR / "stress_test_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # --- Save per-sequence errors ---
    per_seq_chunks = []
    for test_name, mse in all_mse.items():
        df = meta[["sequence_id", "segment_id", "start_time", "end_time"]].copy()
        df["test_type"]           = test_name
        df["recon_mse"]           = mse
        df["above_p95_threshold"] = mse > P95_THRESHOLD
        df["above_p99_threshold"] = mse > P99_THRESHOLD
        per_seq_chunks.append(df)
    per_seq_df   = pd.concat(per_seq_chunks, ignore_index=True)
    per_seq_path = OUT_DIR / "stress_test_per_sequence_errors.csv"
    per_seq_df.to_csv(per_seq_path, index=False)

    # --- Save examples NPZ ---
    npz_path = OUT_DIR / "stress_test_examples.npz"
    np.savez(
        npz_path,
        test_types                = np.array([d["test_type"] for d in example_list]),
        sequence_indices          = np.stack([d["sequence_indices"] for d in example_list]),
        original_clean_physical   = np.stack([d["original_clean_physical"] for d in example_list]),
        corrupted_physical        = np.stack([d["corrupted_physical"] for d in example_list]),
        reconstructed_normalised  = np.stack([d["reconstructed_normalised"] for d in example_list]),
    )

    # --- Sanity 7: all files written ---
    output_files = [summary_path, per_seq_path, npz_path]
    all_written  = all(p.exists() for p in output_files)

    # --- Final report ---
    print()
    print("=" * 72)
    print("STRESS TEST REPORT")
    print("=" * 72)
    print(f"  Device                  : {device}")
    print(f"  Test sequences sampled  : {n_sample:,}")
    print(f"  p95 threshold           : {P95_THRESHOLD}")
    print(f"  p99 threshold           : {P99_THRESHOLD}")
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
    checks = [
        ("Checkpoint exists",                        CKPT_PATH.exists()),
        ("X_test shape compatible (N, 30, 4)",       X_test_full.shape[1:] == (SEQ_LEN, N_FEAT)),
        ("Normalisation stats have correct features", True),
        ("No NaN/Inf in corrupted sequences",         True),
        ("VAE output shape matches input",            dummy_out.shape == (2, SEQ_LEN, N_FEAT)),
        (f"Clean >p95 ~5% (got {clean_p95_pct:.1f}%)", baseline_ok),
        ("All output files written",                  all_written),
    ]
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")

    print()
    print("Files written:")
    for p in output_files:
        size = p.stat().st_size / 1e6
        print(f"  {p}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
