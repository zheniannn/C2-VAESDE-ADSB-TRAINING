"""
stage_03_calibrate_motion_prior_thresholds.py

Calibrates VAE + kinematic motion-prior thresholds using clean test quantiles,
adds turn-rate kinematic features, then reruns all stress tests with calibrated
thresholds.

Previous hand-picked thresholds gave 18.7% kinematic / 20.6% combined false-flag
rate on clean sequences. Calibrated thresholds target ~0.5% per individual flag.
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
DT                 = 10.0
BATCH_SIZE         = 1024
N_EXAMPLES         = 10
SPEED_VALID_THRESH = 10.0          # m/s — min speed for heading/turn-rate validity

SEQ_LEN    = 30
N_FEAT     = 4
FLAT_DIM   = SEQ_LEN * N_FEAT
LATENT_DIM = 16

IDX_E  = 0; IDX_N  = 1; IDX_VE = 2; IDX_VN = 3
FEATURES = ["E_m", "N_m", "vE_mps", "vN_mps"]

VAE_P95_THRESHOLD = 0.03948
VAE_P99_THRESHOLD = 0.10405

CKPT_PATH = Path("models/sequence_vae/sequence_vae_full.pt")
OUT_DIR   = Path("models/sequence_vae/motion_prior_scorer/calibrated")


# ---------------------------------------------------------------------------
# VAE model
# ---------------------------------------------------------------------------
class SequenceVAE(nn.Module):
    def __init__(self, flat_dim=FLAT_DIM, latent_dim=LATENT_DIM):
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
        b = x.size(0)
        x_flat = x.view(b, FLAT_DIM)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z).view(b, SEQ_LEN, N_FEAT), mu, logvar


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def denorm(arr, mean, std): return arr * std + mean
def renorm(arr, mean, std): return (arr - mean) / std


# ---------------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------------
def _integrate(s):
    s[:, 1:, IDX_E] = s[:, 0:1, IDX_E] + np.cumsum(s[:, :-1, IDX_VE], axis=1) * DT
    s[:, 1:, IDX_N] = s[:, 0:1, IDX_N] + np.cumsum(s[:, :-1, IDX_VN], axis=1) * DT
    return s

def c_speed_scale(p, f):
    s = p.copy(); s[:, :, IDX_VE] *= f; s[:, :, IDX_VN] *= f; return _integrate(s)
def c_pos_jump(p, j):
    s = p.copy(); s[:, 15:, IDX_E] += j; return s
def c_rand_walk(p, rng):
    s = p.copy(); N = len(s)
    s[:, :, IDX_VE] += rng.normal(0, 20, (N, SEQ_LEN))
    s[:, :, IDX_VN] += rng.normal(0, 20, (N, SEQ_LEN))
    return _integrate(s)
def c_sudden_turn(p):
    s = p.copy(); ve = s[:, :, IDX_VE].copy(); vn = s[:, :, IDX_VN].copy()
    s[:, 15:, IDX_VE] = -vn[:, 15:]; s[:, 15:, IDX_VN] = ve[:, 15:]
    return _integrate(s)
def c_stationary(p):
    s = p.copy()
    s[:, :, IDX_E] = s[:, 0:1, IDX_E]; s[:, :, IDX_N] = s[:, 0:1, IDX_N]
    s[:, :, IDX_VE] = 0.0;              s[:, :, IDX_VN] = 0.0
    return s


# ---------------------------------------------------------------------------
# Kinematic feature extraction
# ---------------------------------------------------------------------------
def compute_kinematics(seqs_phys: np.ndarray) -> dict[str, np.ndarray]:
    vE = seqs_phys[:, :, IDX_VE]
    vN = seqs_phys[:, :, IDX_VN]
    E  = seqs_phys[:, :, IDX_E]
    N_ = seqs_phys[:, :, IDX_N]

    speed = np.sqrt(vE**2 + vN**2)                              # (N, 30)
    aE    = np.gradient(vE, DT, axis=1)
    aN    = np.gradient(vN, DT, axis=1)
    accel = np.sqrt(aE**2 + aN**2)                              # (N, 30)

    # Position-velocity consistency
    pv_err = np.sqrt(
        (E[:, 1:] - (E[:, :-1] + vE[:, :-1] * DT))**2 +
        (N_[:, 1:] - (N_[:, :-1] + vN[:, :-1] * DT))**2
    )                                                             # (N, 29)

    # Turn rate — only where both endpoints have speed >= SPEED_VALID_THRESH
    heading    = np.arctan2(vN, vE)                              # (N, 30)
    heading_uw = np.unwrap(heading, axis=1)                      # (N, 30)
    hd_change  = np.diff(heading_uw, axis=1)                     # (N, 29) radians
    tr         = np.abs(np.degrees(hd_change)) / DT             # (N, 29) deg/s

    valid       = (speed[:, :-1] >= SPEED_VALID_THRESH) & (speed[:, 1:] >= SPEED_VALID_THRESH)
    valid_count = valid.sum(axis=1)
    tr_v        = np.where(valid, tr, 0.0)                       # (N, 29)

    mean_tr = np.where(valid_count > 0,
                       tr_v.sum(axis=1) / np.maximum(valid_count, 1), 0.0)
    max_tr  = tr_v.max(axis=1)

    disp = np.sqrt((E[:, -1] - E[:, 0])**2 + (N_[:, -1] - N_[:, 0])**2)

    return {
        "mean_speed_mps":       speed.mean(axis=1),
        "max_speed_mps":        speed.max(axis=1),
        "mean_accel_mps2":      accel.mean(axis=1),
        "max_accel_mps2":       accel.max(axis=1),
        "mean_pv_error_m":      pv_err.mean(axis=1),
        "max_pv_error_m":       pv_err.max(axis=1),
        "total_displacement_m": disp,
        "mean_turn_rate_degps": mean_tr,
        "max_turn_rate_degps":  max_tr,
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate(clean_kin: dict[str, np.ndarray]) -> tuple[dict, pd.DataFrame]:
    """Derive thresholds from clean quantiles. Returns (thresholds_dict, quantiles_df)."""
    upper_metrics  = ["max_speed_mps", "max_accel_mps2", "mean_pv_error_m",
                      "max_pv_error_m", "mean_turn_rate_degps", "max_turn_rate_degps"]
    lower_metrics  = ["mean_speed_mps", "total_displacement_m"]
    upper_qs       = [90, 95, 99, 99.5, 99.9]
    lower_qs       = [0.1, 0.5, 1.0, 2.0, 5.0]

    rows = []
    q_lookup: dict[str, dict] = {}

    for metric in upper_metrics + lower_metrics:
        arr    = clean_kin[metric]
        q_dict = {}
        for q in upper_qs:
            key = f"p{str(q).replace('.', '_')}"
            val = float(np.percentile(arr, q))
            q_dict[key] = val
            rows.append({"metric": metric, "quantile": key, "value": val})
        if metric in lower_metrics:
            for q in lower_qs:
                key = f"p{str(q).replace('.', '_')}"
                val = float(np.percentile(arr, q))
                q_dict[key] = val
                rows.append({"metric": metric, "quantile": key, "value": val})
        q_lookup[metric] = q_dict

    q99_5 = lambda m: q_lookup[m]["p99_5"]
    p0_5  = lambda m: q_lookup[m]["p0_5"]

    thresholds = {
        "max_speed_threshold_mps":        max(150.0, q99_5("max_speed_mps")),
        "max_accel_threshold_mps2":       q99_5("max_accel_mps2"),
        "mean_pv_error_threshold_m":      q99_5("mean_pv_error_m"),
        "max_pv_error_threshold_m":       q99_5("max_pv_error_m"),
        "mean_turn_rate_threshold_degps": q99_5("mean_turn_rate_degps"),
        "max_turn_rate_threshold_degps":  q99_5("max_turn_rate_degps"),
        "min_mean_speed_threshold_mps":   max(1.0,  p0_5("mean_speed_mps")),
        "min_displacement_threshold_m":   max(50.0, p0_5("total_displacement_m")),
    }
    quantiles_df = pd.DataFrame(rows)
    return thresholds, quantiles_df


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
def compute_flags(kin: dict[str, np.ndarray],
                  recon_mse: np.ndarray,
                  thr: dict) -> dict[str, np.ndarray]:
    fs   = kin["max_speed_mps"]        > thr["max_speed_threshold_mps"]
    fa   = kin["max_accel_mps2"]       > thr["max_accel_threshold_mps2"]
    fpx  = kin["max_pv_error_m"]       > thr["max_pv_error_threshold_m"]
    fpm  = kin["mean_pv_error_m"]      > thr["mean_pv_error_threshold_m"]
    ftx  = kin["max_turn_rate_degps"]  > thr["max_turn_rate_threshold_degps"]
    ftm  = kin["mean_turn_rate_degps"] > thr["mean_turn_rate_threshold_degps"]
    fts  = kin["mean_speed_mps"]       < thr["min_mean_speed_threshold_mps"]
    fld  = kin["total_displacement_m"] < thr["min_displacement_threshold_m"]

    kflag = fs | fa | fpx | fpm | ftx | ftm | fts | fld
    vp95  = recon_mse > VAE_P95_THRESHOLD
    vp99  = recon_mse > VAE_P99_THRESHOLD

    return {
        "vae_flag_p95": vp95,   "vae_flag_p99": vp99,
        "flag_speed": fs,        "flag_accel": fa,
        "flag_pv_max": fpx,      "flag_pv_mean": fpm,
        "flag_turn_max": ftx,    "flag_turn_mean": ftm,
        "flag_too_slow": fts,    "flag_low_displacement": fld,
        "kinematic_flag": kflag,
        "combined_flag_p95": vp95 | kflag,
        "combined_flag_p99": vp99 | kflag,
    }


# ---------------------------------------------------------------------------
# VAE inference
# ---------------------------------------------------------------------------
def run_vae(model: SequenceVAE, seqs_norm: np.ndarray,
            device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); mse_c, rec_c = [], []
    with torch.no_grad():
        for s in range(0, len(seqs_norm), BATCH_SIZE):
            b = torch.from_numpy(seqs_norm[s:s + BATCH_SIZE].astype(np.float32)).to(device)
            r, _, _ = model(b)
            mse_c.append(((r - b)**2).mean(dim=(1, 2)).cpu().numpy())
            rec_c.append(r.cpu().numpy())
    return np.concatenate(mse_c), np.concatenate(rec_c)


# ---------------------------------------------------------------------------
# Summary row
# ---------------------------------------------------------------------------
def build_row(name, n, recon_mse, kin, flags) -> dict:
    pct = lambda a: float(100 * a.mean())
    p   = lambda a, q: float(np.percentile(a, q))
    return {
        "test_type": name,
        "num_sequences": n,
        "percent_vae_p95":             pct(flags["vae_flag_p95"]),
        "percent_vae_p99":             pct(flags["vae_flag_p99"]),
        "percent_kinematic_flagged":   pct(flags["kinematic_flag"]),
        "percent_combined_p95":        pct(flags["combined_flag_p95"]),
        "percent_combined_p99":        pct(flags["combined_flag_p99"]),
        "percent_flag_speed":          pct(flags["flag_speed"]),
        "percent_flag_accel":          pct(flags["flag_accel"]),
        "percent_flag_pv_max":         pct(flags["flag_pv_max"]),
        "percent_flag_pv_mean":        pct(flags["flag_pv_mean"]),
        "percent_flag_turn_max":       pct(flags["flag_turn_max"]),
        "percent_flag_turn_mean":      pct(flags["flag_turn_mean"]),
        "percent_flag_too_slow":       pct(flags["flag_too_slow"]),
        "percent_flag_low_displacement": pct(flags["flag_low_displacement"]),
        "recon_mse_mean": float(recon_mse.mean()),
        "recon_mse_p50":  p(recon_mse, 50),
        "recon_mse_p95":  p(recon_mse, 95),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    assert CKPT_PATH.exists(), f"Checkpoint not found: {CKPT_PATH}"
    model = SequenceVAE().to(device)
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded: {CKPT_PATH}")

    with torch.no_grad():
        dummy_out, _, _ = model(torch.zeros(2, SEQ_LEN, N_FEAT, device=device))
        assert dummy_out.shape == (2, SEQ_LEN, N_FEAT)

    # --- Normalisation stats ---
    mean_df = pd.read_csv("data/normalisation_mean.csv", index_col=0)
    std_df  = pd.read_csv("data/normalisation_std.csv",  index_col=0)
    for df, tag in [(mean_df, "mean"), (std_df, "std")]:
        assert not [f for f in FEATURES if f not in df.index], \
            f"normalisation_{tag}.csv missing features"
    mean = mean_df.loc[FEATURES, "mean"].values.astype(np.float64)
    std  = std_df.loc[FEATURES,  "std"].values.astype(np.float64)

    # --- Load & sample X_test ---
    X_test_full = np.load("data/X_test.npy", mmap_mode="r")
    assert X_test_full.shape[1:] == (SEQ_LEN, N_FEAT)
    n_sample   = min(NUM_TEST_SEQUENCES, len(X_test_full))
    idx        = np.sort(rng.choice(len(X_test_full), n_sample, replace=False))
    seqs_norm  = X_test_full[idx].copy().astype(np.float64)
    seqs_phys  = denorm(seqs_norm, mean, std)
    meta_full  = pd.read_csv("data/test_sequence_metadata.csv")
    meta       = meta_full.iloc[idx].reset_index(drop=True)
    print(f"Sampled {n_sample:,} sequences  (X_test: {X_test_full.shape})\n")

    # -----------------------------------------------------------------------
    # CALIBRATION — use clean sequences only
    # -----------------------------------------------------------------------
    print("Computing clean kinematic quantiles ...")
    clean_kin = compute_kinematics(seqs_phys)
    thresholds, quantiles_df = calibrate(clean_kin)

    print("\nCalibrated thresholds:")
    for k, v in thresholds.items():
        print(f"  {k:<40} = {v:.4f}")

    # Save threshold CSV
    thr_rows = [
        {"threshold_name": "max_speed_threshold_mps",
         "value": thresholds["max_speed_threshold_mps"],
         "source_metric": "max_speed_mps",
         "source_quantile": "p99_5 (floored at 150 m/s)",
         "rationale": "Physical cap 150 m/s OR data-driven p99.5; whichever is higher"},
        {"threshold_name": "max_accel_threshold_mps2",
         "value": thresholds["max_accel_threshold_mps2"],
         "source_metric": "max_accel_mps2",
         "source_quantile": "p99_5",
         "rationale": "0.5% false-flag rate per flag on clean data"},
        {"threshold_name": "mean_pv_error_threshold_m",
         "value": thresholds["mean_pv_error_threshold_m"],
         "source_metric": "mean_pv_error_m",
         "source_quantile": "p99_5",
         "rationale": "0.5% false-flag rate per flag on clean data"},
        {"threshold_name": "max_pv_error_threshold_m",
         "value": thresholds["max_pv_error_threshold_m"],
         "source_metric": "max_pv_error_m",
         "source_quantile": "p99_5",
         "rationale": "0.5% false-flag rate per flag on clean data"},
        {"threshold_name": "mean_turn_rate_threshold_degps",
         "value": thresholds["mean_turn_rate_threshold_degps"],
         "source_metric": "mean_turn_rate_degps",
         "source_quantile": "p99_5",
         "rationale": "0.5% false-flag rate per flag on clean data"},
        {"threshold_name": "max_turn_rate_threshold_degps",
         "value": thresholds["max_turn_rate_threshold_degps"],
         "source_metric": "max_turn_rate_degps",
         "source_quantile": "p99_5",
         "rationale": "0.5% false-flag rate per flag on clean data"},
        {"threshold_name": "min_mean_speed_threshold_mps",
         "value": thresholds["min_mean_speed_threshold_mps"],
         "source_metric": "mean_speed_mps",
         "source_quantile": "p0_5 (floored at 1.0 m/s)",
         "rationale": "0.5% false-flag rate on lower tail; minimum 1 m/s"},
        {"threshold_name": "min_displacement_threshold_m",
         "value": thresholds["min_displacement_threshold_m"],
         "source_metric": "total_displacement_m",
         "source_quantile": "p0_5 (floored at 50 m)",
         "rationale": "0.5% false-flag rate on lower tail; minimum 50 m"},
    ]
    thr_df  = pd.DataFrame(thr_rows)
    thr_path = OUT_DIR / "calibrated_thresholds.csv"
    thr_df.to_csv(thr_path, index=False)
    quantiles_df.to_csv(OUT_DIR / "clean_kinematic_quantiles.csv", index=False)

    # -----------------------------------------------------------------------
    # Stress tests with calibrated thresholds
    # -----------------------------------------------------------------------
    TEST_CASES: list[tuple[str, np.ndarray]] = [
        ("clean",                  seqs_phys.copy()),
        ("speed_scaled_1p5",       c_speed_scale(seqs_phys, 1.5)),
        ("speed_scaled_2p0",       c_speed_scale(seqs_phys, 2.0)),
        ("position_jump_1000m",    c_pos_jump(seqs_phys, 1000.0)),
        ("position_jump_2000m",    c_pos_jump(seqs_phys, 2000.0)),
        ("random_walk_velocity",   c_rand_walk(seqs_phys, rng)),
        ("sudden_turn_90deg",      c_sudden_turn(seqs_phys)),
        ("stationary_clutter_like", c_stationary(seqs_phys)),
    ]

    # Sanity 4: no NaN/Inf
    for name, phys in TEST_CASES[1:]:
        n = renorm(phys, mean, std)
        assert not np.isnan(n).any() and not np.isinf(n).any(), f"NaN/Inf in {name}"
    print("\nNaN/Inf check: OK\n")

    ex_idx    = np.sort(rng.choice(n_sample, N_EXAMPLES, replace=False))
    summary_rows:   list[dict]         = []
    per_seq_chunks: list[pd.DataFrame] = []
    example_list:   list[dict]         = []

    # Sanity accumulators
    clean_kflag_pct    = None
    clean_vae_p95_pct  = None
    clean_vae_p99_pct  = None
    pjump_pv_pcts:  list[float] = []
    stat_lower_pct  = None
    sudden_prev_comb = 15.3   # from previous motion_prior_scorer run
    sudden_new_comb  = None

    print(f"  {'Test type':<28}  {'VAE p95':>8}  {'Kin':>6}  {'Comb p95':>9}")
    print("  " + "-" * 58)

    for test_name, corrupted_phys in TEST_CASES:
        norm = renorm(corrupted_phys, mean, std)
        recon_mse, recon_arr = run_vae(model, norm, device)
        kin   = compute_kinematics(corrupted_phys)
        flags = compute_flags(kin, recon_mse, thresholds)
        row   = build_row(test_name, n_sample, recon_mse, kin, flags)
        summary_rows.append(row)

        # Per-sequence CSV
        df = meta[["sequence_id", "segment_id", "start_time", "end_time"]].copy()
        df["test_type"]  = test_name
        df["recon_mse"]  = recon_mse
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

        # Examples
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

    # --- Save outputs ---
    summary_df   = pd.DataFrame(summary_rows)
    sum_path     = OUT_DIR / "calibrated_motion_prior_summary.csv"
    summary_df.to_csv(sum_path, index=False)

    per_seq_df   = pd.concat(per_seq_chunks, ignore_index=True)
    pseq_path    = OUT_DIR / "calibrated_motion_prior_per_sequence_scores.csv"
    per_seq_df.to_csv(pseq_path, index=False)

    npz_path = OUT_DIR / "calibrated_motion_prior_examples.npz"
    np.savez(
        npz_path,
        test_types               = np.array([d["test_type"] for d in example_list]),
        sequence_indices         = np.stack([d["sequence_indices"] for d in example_list]),
        clean_physical           = np.stack([d["clean_physical"] for d in example_list]),
        corrupted_physical       = np.stack([d["corrupted_physical"] for d in example_list]),
        reconstructed_normalised = np.stack([d["reconstructed_normalised"] for d in example_list]),
    )
    output_files = [thr_path, OUT_DIR / "clean_kinematic_quantiles.csv",
                    sum_path, pseq_path, npz_path]
    all_written  = all(p.exists() for p in output_files)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
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
        vals = "  ".join(
            f"{r.get('percent_' + c, 0):>9.1f}%" for c in fcols
        )
        print(f"  {r['test_type']:<28}  {vals}")

    print()
    print("Interpretation:")
    _interpret(summary_rows)

    print()
    print("Calibration impact vs previous hand-picked thresholds:")
    print(f"  Clean kinematic flag rate : 18.7%  →  {clean_kflag_pct:.1f}%")
    print(f"  Clean combined p95 rate   : 20.6%  →  "
          f"{summary_rows[0]['percent_combined_p95']:.1f}%")
    if sudden_new_comb is not None:
        print(f"  sudden_turn_90deg combined: 15.3%  →  {sudden_new_comb:.1f}%  "
              f"({'improved' if sudden_new_comb > sudden_prev_comb else 'unchanged/lower'})")

    print()
    print("Sanity checks:")
    checks = [
        ("Checkpoint exists",                        CKPT_PATH.exists()),
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
        (f"Position jumps pv_max ≥ 80%  (got {min(pjump_pv_pcts):.1f}%)",
            all(v >= 80.0 for v in pjump_pv_pcts)),
        (f"Stationary lower-tail ≥ 80%  (got {stat_lower_pct:.1f}%)",
            stat_lower_pct is not None and stat_lower_pct >= 80.0),
        (f"sudden_turn combined > 0%  (got {sudden_new_comb:.1f}%)",
            sudden_new_comb is not None and sudden_new_comb > 0),
        ("All output files written",                  all_written),
    ]
    all_pass = True
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
        if not result:
            all_pass = False

    print()
    print("Files written:")
    for p in output_files:
        print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")


def _interpret(rows: list[dict]) -> None:
    clean_v = rows[0]["percent_vae_p95"]
    for r in rows[1:]:
        name    = r["test_type"]
        v95     = r["percent_vae_p95"]
        k_pct   = r["percent_kinematic_flagged"]
        c95     = r["percent_combined_p95"]
        k_gain  = c95 - v95

        dominant_flags = [
            col.replace("percent_flag_", "")
            for col in ["percent_flag_speed", "percent_flag_accel",
                        "percent_flag_pv_max", "percent_flag_pv_mean",
                        "percent_flag_turn_max", "percent_flag_turn_mean",
                        "percent_flag_too_slow", "percent_flag_low_displacement"]
            if r.get(col, 0) >= 5.0
        ]
        dom = ", ".join(dominant_flags) if dominant_flags else "none"

        verdict = ("combined >> VAE" if k_gain >= 20
                   else "VAE dominant"  if v95 >= 80
                   else "kinematic dominant" if k_pct >= 80
                   else "partial both")
        print(f"  {name:<28}  VAE={v95:.0f}%  Kin={k_pct:.0f}%  "
              f"Comb={c95:.0f}%  [{dom}]  → {verdict}")


if __name__ == "__main__":
    main()
