"""
stage_04_motion_prior_scorer.py

Combines VAE reconstruction scoring with explicit kinematic consistency checks.
Compares VAE-only, kinematic-only, and combined detection across eight stress-test
corruption categories.

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
DT                 = 10.0          # seconds between ADS-B pings
BATCH_SIZE         = 1024
N_EXAMPLES         = 10

SEQ_LEN    = 30
N_FEAT     = 4
FLAT_DIM   = SEQ_LEN * N_FEAT      # 120
LATENT_DIM = 16

IDX_E  = 0   # E_m
IDX_N  = 1   # N_m
IDX_VE = 2   # vE_mps
IDX_VN = 3   # vN_mps
FEATURES = ["E_m", "N_m", "vE_mps", "vN_mps"]

# VAE thresholds (from full-model training)
VAE_P95_THRESHOLD = 0.03948
VAE_P99_THRESHOLD = 0.10405

# Kinematic thresholds
MAX_SPEED_THRESHOLD      = 150.0    # m/s
MAX_ACCEL_THRESHOLD      = 10.0     # m/s²
MAX_PV_ERROR_THRESHOLD   = 500.0    # m
MEAN_PV_ERROR_THRESHOLD  = 150.0    # m
MIN_MEAN_SPEED_THRESHOLD = 5.0      # m/s
MIN_DISPLACEMENT_THRESHOLD = 100.0  # m

CKPT_PATH = Path("models/sequence_vae/sequence_vae_full.pt")
OUT_DIR   = Path("models/sequence_vae/motion_prior_scorer")


# ---------------------------------------------------------------------------
# VAE model  (identical to stage_01_train_sequence_vae.py)
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

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterise(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        b           = x.size(0)
        x_flat      = x.view(b, FLAT_DIM)
        mu, logvar  = self.encode(x_flat)
        z           = self.reparameterise(mu, logvar)
        recon_flat  = self.decoder(z)
        return recon_flat.view(b, SEQ_LEN, N_FEAT), mu, logvar


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def denormalise(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return arr * std + mean

def renormalise(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (arr - mean) / std


# ---------------------------------------------------------------------------
# Corruption functions  (physical-unit in → physical-unit out)
# ---------------------------------------------------------------------------
def _integrate(s: np.ndarray) -> np.ndarray:
    """Rebuild E_m and N_m from velocities, keeping position[0] fixed."""
    s[:, 1:, IDX_E] = s[:, 0:1, IDX_E] + np.cumsum(s[:, :-1, IDX_VE], axis=1) * DT
    s[:, 1:, IDX_N] = s[:, 0:1, IDX_N] + np.cumsum(s[:, :-1, IDX_VN], axis=1) * DT
    return s

def corrupt_speed_scale(p: np.ndarray, factor: float) -> np.ndarray:
    s = p.copy(); s[:, :, IDX_VE] *= factor; s[:, :, IDX_VN] *= factor
    return _integrate(s)

def corrupt_position_jump(p: np.ndarray, jump_m: float) -> np.ndarray:
    s = p.copy(); s[:, 15:, IDX_E] += jump_m; return s

def corrupt_random_walk_velocity(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    s = p.copy()
    s[:, :, IDX_VE] += rng.normal(0.0, 20.0, (len(s), SEQ_LEN))
    s[:, :, IDX_VN] += rng.normal(0.0, 20.0, (len(s), SEQ_LEN))
    return _integrate(s)

def corrupt_sudden_turn(p: np.ndarray) -> np.ndarray:
    s = p.copy()
    vE_o = s[:, :, IDX_VE].copy(); vN_o = s[:, :, IDX_VN].copy()
    s[:, 15:, IDX_VE] = -vN_o[:, 15:]; s[:, 15:, IDX_VN] = vE_o[:, 15:]
    return _integrate(s)

def corrupt_stationary(p: np.ndarray) -> np.ndarray:
    s = p.copy()
    s[:, :, IDX_E] = s[:, 0:1, IDX_E]; s[:, :, IDX_N] = s[:, 0:1, IDX_N]
    s[:, :, IDX_VE] = 0.0;             s[:, :, IDX_VN] = 0.0
    return s


# ---------------------------------------------------------------------------
# Kinematic feature extraction
# ---------------------------------------------------------------------------
def compute_kinematics(seqs_phys: np.ndarray) -> dict[str, np.ndarray]:
    """Returns per-sequence kinematic scalars, each shape (N,)."""
    E  = seqs_phys[:, :, IDX_E]
    N_ = seqs_phys[:, :, IDX_N]
    vE = seqs_phys[:, :, IDX_VE]
    vN = seqs_phys[:, :, IDX_VN]

    speed  = np.sqrt(vE ** 2 + vN ** 2)                           # (N, 30)
    aE     = np.gradient(vE, DT, axis=1)
    aN     = np.gradient(vN, DT, axis=1)
    accel  = np.sqrt(aE ** 2 + aN ** 2)                           # (N, 30)

    # Position-velocity consistency: how well velocity predicts the next position
    pred_E = E[:, :-1] + vE[:, :-1] * DT                         # (N, 29)
    pred_N = N_[:, :-1] + vN[:, :-1] * DT
    pv_err = np.sqrt((E[:, 1:] - pred_E)**2 + (N_[:, 1:] - pred_N)**2)  # (N, 29)

    displacement = np.sqrt((E[:, -1] - E[:, 0])**2 + (N_[:, -1] - N_[:, 0])**2)

    return {
        "mean_speed_mps":       speed.mean(axis=1),
        "max_speed_mps":        speed.max(axis=1),
        "mean_accel_mps2":      accel.mean(axis=1),
        "max_accel_mps2":       accel.max(axis=1),
        "mean_pv_error_m":      pv_err.mean(axis=1),
        "max_pv_error_m":       pv_err.max(axis=1),
        "total_displacement_m": displacement,
    }


# ---------------------------------------------------------------------------
# Kinematic + VAE flags
# ---------------------------------------------------------------------------
def compute_flags(kin: dict[str, np.ndarray],
                  recon_mse: np.ndarray) -> dict[str, np.ndarray]:
    fs  = kin["max_speed_mps"]       > MAX_SPEED_THRESHOLD
    fa  = kin["max_accel_mps2"]      > MAX_ACCEL_THRESHOLD
    fpx = kin["max_pv_error_m"]      > MAX_PV_ERROR_THRESHOLD
    fpm = kin["mean_pv_error_m"]     > MEAN_PV_ERROR_THRESHOLD
    fts = kin["mean_speed_mps"]      < MIN_MEAN_SPEED_THRESHOLD
    fld = kin["total_displacement_m"] < MIN_DISPLACEMENT_THRESHOLD

    kflag = fs | fa | fpx | fpm | fts | fld

    vp95 = recon_mse > VAE_P95_THRESHOLD
    vp99 = recon_mse > VAE_P99_THRESHOLD

    return {
        "vae_flag_p95":         vp95,
        "vae_flag_p99":         vp99,
        "flag_speed":           fs,
        "flag_accel":           fa,
        "flag_pv_max":          fpx,
        "flag_pv_mean":         fpm,
        "flag_too_slow":        fts,
        "flag_low_displacement": fld,
        "kinematic_flag":       kflag,
        "combined_flag_p95":    vp95 | kflag,
        "combined_flag_p99":    vp99 | kflag,
    }


# ---------------------------------------------------------------------------
# VAE inference
# ---------------------------------------------------------------------------
def compute_recon_mse(model: SequenceVAE,
                      seqs_norm: np.ndarray,
                      device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mse shape (N,), reconstructions shape (N, 30, 4))."""
    model.eval()
    mse_chunks, recon_chunks = [], []
    with torch.no_grad():
        for s in range(0, len(seqs_norm), BATCH_SIZE):
            b_np  = seqs_norm[s:s + BATCH_SIZE].astype(np.float32)
            b     = torch.from_numpy(b_np).to(device)
            recon, _, _ = model(b)
            mse_chunks.append(((recon - b) ** 2).mean(dim=(1, 2)).cpu().numpy())
            recon_chunks.append(recon.cpu().numpy())
    return np.concatenate(mse_chunks), np.concatenate(recon_chunks, axis=0)


# ---------------------------------------------------------------------------
# Summary row builder
# ---------------------------------------------------------------------------
def build_summary_row(test_type: str,
                      recon_mse: np.ndarray,
                      kin: dict[str, np.ndarray],
                      flags: dict[str, np.ndarray]) -> dict:
    pct = lambda arr: float(100 * arr.mean())
    p   = lambda arr, q: float(np.percentile(arr, q))

    return {
        "test_type": test_type,
        # VAE-only
        "percent_vae_above_p95":          pct(flags["vae_flag_p95"]),
        "percent_vae_above_p99":          pct(flags["vae_flag_p99"]),
        # Kinematic-only
        "percent_kinematic_flagged":       pct(flags["kinematic_flag"]),
        "percent_flag_speed":              pct(flags["flag_speed"]),
        "percent_flag_accel":              pct(flags["flag_accel"]),
        "percent_flag_pv_max":             pct(flags["flag_pv_max"]),
        "percent_flag_pv_mean":            pct(flags["flag_pv_mean"]),
        "percent_flag_too_slow":           pct(flags["flag_too_slow"]),
        "percent_flag_low_displacement":   pct(flags["flag_low_displacement"]),
        # Combined
        "percent_combined_p95_flagged":    pct(flags["combined_flag_p95"]),
        "percent_combined_p99_flagged":    pct(flags["combined_flag_p99"]),
        # Recon MSE distribution
        "recon_mse_mean": float(recon_mse.mean()),
        "recon_mse_p50":  p(recon_mse, 50),
        "recon_mse_p95":  p(recon_mse, 95),
        "recon_mse_p99":  p(recon_mse, 99),
        # Speed
        "max_speed_mean": float(kin["max_speed_mps"].mean()),
        "max_speed_p95":  p(kin["max_speed_mps"], 95),
        "max_speed_p99":  p(kin["max_speed_mps"], 99),
        "max_speed_max":  float(kin["max_speed_mps"].max()),
        # Accel
        "max_accel_mean": float(kin["max_accel_mps2"].mean()),
        "max_accel_p95":  p(kin["max_accel_mps2"], 95),
        "max_accel_p99":  p(kin["max_accel_mps2"], 99),
        "max_accel_max":  float(kin["max_accel_mps2"].max()),
        # PV error (max per sequence)
        "max_pv_error_mean": float(kin["max_pv_error_m"].mean()),
        "max_pv_error_p95":  p(kin["max_pv_error_m"], 95),
        "max_pv_error_p99":  p(kin["max_pv_error_m"], 99),
        "max_pv_error_max":  float(kin["max_pv_error_m"].max()),
        # PV error (mean per sequence)
        "mean_pv_error_mean": float(kin["mean_pv_error_m"].mean()),
        "mean_pv_error_p95":  p(kin["mean_pv_error_m"], 95),
        "mean_pv_error_p99":  p(kin["mean_pv_error_m"], 99),
        "mean_pv_error_max":  float(kin["mean_pv_error_m"].max()),
        # Displacement
        "displacement_mean": float(kin["total_displacement_m"].mean()),
        "displacement_p5":   p(kin["total_displacement_m"],  5),
        "displacement_p50":  p(kin["total_displacement_m"], 50),
        "displacement_p95":  p(kin["total_displacement_m"], 95),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Sanity 1: checkpoint ---
    assert CKPT_PATH.exists(), f"Checkpoint not found: {CKPT_PATH}"
    model = SequenceVAE(FLAT_DIM, LATENT_DIM).to(device)
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model from {CKPT_PATH}")

    # --- Sanity 5: output shape ---
    with torch.no_grad():
        dummy_out, _, _ = model(torch.zeros(2, SEQ_LEN, N_FEAT, device=device))
        assert dummy_out.shape == (2, SEQ_LEN, N_FEAT)

    # --- Normalisation stats ---
    mean_df = pd.read_csv("data/normalisation_mean.csv", index_col=0)
    std_df  = pd.read_csv("data/normalisation_std.csv",  index_col=0)

    # --- Sanity 3 ---
    for df, tag in [(mean_df, "mean"), (std_df, "std")]:
        missing = [f for f in FEATURES if f not in df.index]
        assert not missing, f"normalisation_{tag}.csv missing: {missing}"

    mean = mean_df.loc[FEATURES, "mean"].values.astype(np.float64)
    std  = std_df.loc[FEATURES,  "std"].values.astype(np.float64)

    # --- Load X_test ---
    X_test_full = np.load("data/X_test.npy", mmap_mode="r")
    print(f"X_test shape: {X_test_full.shape}")

    # --- Sanity 2 ---
    assert X_test_full.ndim == 3 and X_test_full.shape[1:] == (SEQ_LEN, N_FEAT)

    # --- Sample sequences ---
    n_avail  = len(X_test_full)
    n_sample = min(NUM_TEST_SEQUENCES, n_avail)
    idx      = np.sort(rng.choice(n_avail, n_sample, replace=False))
    seqs_norm = X_test_full[idx].copy().astype(np.float64)
    seqs_phys = denormalise(seqs_norm, mean, std)
    print(f"Sampled {n_sample:,} sequences\n")

    meta_full = pd.read_csv("data/test_sequence_metadata.csv")
    meta      = meta_full.iloc[idx].reset_index(drop=True)

    # --- Define all test cases: (name, corrupted_physical_array) ---
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

    # --- Sanity 4: no NaN/Inf in renormalised arrays ---
    for name, phys in TEST_CASES[1:]:
        norm = renormalise(phys, mean, std)
        assert not np.isnan(norm).any() and not np.isinf(norm).any(), \
            f"NaN/Inf in {name}"
    print("NaN/Inf check: OK\n")

    # Fixed example indices
    ex_idx = np.sort(rng.choice(n_sample, N_EXAMPLES, replace=False))

    # --- Process each test case ---
    summary_rows:    list[dict]              = []
    per_seq_chunks:  list[pd.DataFrame]      = []
    example_list:    list[dict]              = []

    # Sanity accumulators
    clean_vae_p95_pct  = None
    clean_vae_p99_pct  = None
    pjump_kin_flag_pct: list[float] = []
    stat_kin_flag_pct  = None

    for test_name, corrupted_phys in TEST_CASES:
        norm = renormalise(corrupted_phys, mean, std)

        # VAE
        recon_mse, recon_arr = compute_recon_mse(model, norm, device)

        # Kinematics (in physical space)
        kin   = compute_kinematics(corrupted_phys)
        flags = compute_flags(kin, recon_mse)

        # Summary
        row = build_summary_row(test_name, recon_mse, kin, flags)
        summary_rows.append(row)

        # Per-sequence CSV
        df = meta[["sequence_id", "segment_id", "start_time", "end_time"]].copy()
        df["test_type"]            = test_name
        df["recon_mse"]            = recon_mse
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

        # Examples
        ex_recon = recon_arr[ex_idx]
        example_list.append({
            "test_type":               test_name,
            "sequence_indices":        ex_idx.copy(),
            "clean_physical":          seqs_phys[ex_idx].astype(np.float32),
            "corrupted_physical":      corrupted_phys[ex_idx].astype(np.float32),
            "reconstructed_normalised": ex_recon,
        })

        # Collect sanity accumulators
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

        # Progress line
        print(f"  [{test_name:<28}]  "
              f"VAE_p95={vp95:5.1f}%  "
              f"Kin={kpct:5.1f}%  "
              f"Comb_p95={row['percent_combined_p95_flagged']:5.1f}%")

    # --- Save outputs ---
    summary_df   = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "motion_prior_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    per_seq_df   = pd.concat(per_seq_chunks, ignore_index=True)
    per_seq_path = OUT_DIR / "motion_prior_per_sequence_scores.csv"
    per_seq_df.to_csv(per_seq_path, index=False)

    npz_path = OUT_DIR / "motion_prior_examples.npz"
    np.savez(
        npz_path,
        test_types               = np.array([d["test_type"] for d in example_list]),
        sequence_indices         = np.stack([d["sequence_indices"] for d in example_list]),
        clean_physical           = np.stack([d["clean_physical"] for d in example_list]),
        corrupted_physical       = np.stack([d["corrupted_physical"] for d in example_list]),
        reconstructed_normalised = np.stack([d["reconstructed_normalised"] for d in example_list]),
    )

    output_files = [summary_path, per_seq_path, npz_path]
    all_written  = all(p.exists() for p in output_files)

    # --- Final report ---
    print()
    print("=" * 78)
    print("MOTION PRIOR SCORER REPORT")
    print("=" * 78)
    print(f"  Device                    : {device}")
    print(f"  Test sequences sampled    : {n_sample:,}")
    print(f"  VAE p95 threshold         : {VAE_P95_THRESHOLD}")
    print(f"  VAE p99 threshold         : {VAE_P99_THRESHOLD}")
    print()

    # Table
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
        vals = "  ".join(
            f"{r.get('percent_' + c, 0):>11.1f}%" for c in flag_cols
        )
        print(f"  {r['test_type']:<28}  {vals}")

    print()
    print("Interpretation:")
    _interpret(summary_rows)

    print()
    print("Sanity checks:")
    checks = [
        ("Checkpoint exists",
            CKPT_PATH.exists()),
        ("X_test shape compatible (N, 30, 4)",
            X_test_full.shape[1:] == (SEQ_LEN, N_FEAT)),
        ("Normalisation stats have correct features",
            True),
        ("No NaN/Inf in corrupted sequences",
            True),
        ("VAE output shape matches input",
            dummy_out.shape == (2, SEQ_LEN, N_FEAT)),
        (f"Clean VAE p95 ~5%  (got {clean_vae_p95_pct:.1f}%)",
            3.0 <= clean_vae_p95_pct <= 10.0),
        (f"Clean VAE p99 ~1%  (got {clean_vae_p99_pct:.1f}%)",
            0.5 <= clean_vae_p99_pct <= 3.0),
        (f"Position jumps caught by pv_max  (got {min(pjump_kin_flag_pct):.1f}%)",
            all(v >= 80.0 for v in pjump_kin_flag_pct)),
        (f"Stationary caught by speed/displacement  (got {stat_kin_flag_pct:.1f}%)",
            stat_kin_flag_pct is not None and stat_kin_flag_pct >= 80.0),
        ("All output files written",
            all_written),
    ]
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")

    print()
    print("Files written:")
    for p in output_files:
        print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")


def _interpret(rows: list[dict]) -> None:
    clean_vae = rows[0]["percent_vae_above_p95"]
    for r in rows[1:]:
        name   = r["test_type"]
        v_gain = r["percent_vae_above_p95"] - clean_vae
        k_pct  = r["percent_kinematic_flagged"]
        c_pct  = r["percent_combined_p95_flagged"]
        gain   = c_pct - r["percent_vae_above_p95"]

        # Identify dominant kinematic flags
        flags_hit = [
            col.replace("percent_flag_", "")
            for col in ["percent_flag_speed", "percent_flag_accel",
                        "percent_flag_pv_max", "percent_flag_pv_mean",
                        "percent_flag_too_slow", "percent_flag_low_displacement"]
            if r.get(col, 0) >= 10.0
        ]
        dom = ", ".join(flags_hit) if flags_hit else "none"

        verdict = ("combined adds value" if gain >= 5.0
                   else "VAE alone sufficient" if r["percent_vae_above_p95"] >= 80.0
                   else "kinematic fills gap" if k_pct >= 80.0
                   else "partial detection only")
        print(f"  {name:<28}  VAE+{v_gain:+.0f}%  Kin={k_pct:.0f}%  "
              f"Comb={c_pct:.0f}%  flags=[{dom}]  → {verdict}")


if __name__ == "__main__":
    main()
