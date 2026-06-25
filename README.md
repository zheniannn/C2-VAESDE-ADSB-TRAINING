# VAESDE_model — GA Trajectory Motion Prior

A β-VAE trained on fixed-length ENU trajectory windows from GA aircraft ADS-B data, combined with kinematic consistency flags, to score trajectory sequences as physically plausible or anomalous.

Consumes normalised arrays produced by the **VAESDE_preprocess** repository.

---

## Why this exists

Standard anomaly detectors either flag too aggressively on benign manoeuvres or miss physically implausible trajectories that look statistically normal. This pipeline trains a VAE as a learned motion prior and combines its reconstruction error with explicit kinematic checks to improve coverage across corruption types that neither component catches alone.

---

## Method

Each trajectory is a 30-timestep window of `[E_m, N_m, vE_mps, vN_mps]` (horizontal ENU position and velocity, normalised to zero mean / unit std on the train split).

```
Input (N, 30, 4)
    → flatten → Encoder [Linear(120→256)→ReLU→Linear(256→128)→ReLU] → μ, log σ²  (dim 16)
    → reparameterise → Decoder [Linear(16→128)→ReLU→Linear(128→256)→ReLU→Linear(256→120)]
    → reshape (N, 30, 4)
Loss: MSE_recon + 0.001 × KL
```

A sequence is flagged if its reconstruction MSE exceeds a test-set quantile threshold (p95 or p99), **or** if any kinematic flag fires (speed, acceleration, position-velocity error, turn rate, stationarity). Kinematic thresholds are either hand-picked (`run_score.py`) or calibrated from clean-sequence quantiles at the 0.5 % per-flag false-flag rate (`run_calibrate.py`).

---

## Key result

Detection rates with calibrated VAE + kinematic thresholds (n = 50 000, seed 42):

| Corruption | VAE p95 | Kinematic | Combined p95 |
|---|---:|---:|---:|
| clean (false-flag rate) | 4.5% | 5.5% | **8.9%** |
| speed_scaled_2.0× | 24.3% | 44.1% | 48.6% |
| position_jump_2000 m | 4.5% | **92.5%** | 93.0% |
| random_walk_velocity | **100%** | 47.4% | 100% |
| sudden_turn_90° | 9.1% | 13.2% | 19.5% |
| stationary_clutter | 0% | **100%** | 100% |

The VAE and kinematic flags cover complementary failure modes: the VAE catches velocity-distribution anomalies (random walk) that kinematic thresholds miss, while kinematic flags catch position discontinuities and stationarity that the VAE cannot detect. See [docs/results.md](docs/results.md) for the full breakdown including VAE-only rates and calibrated threshold values.

---

## Quick start

```bash
pip install -e .          # install vaesde package
python3 -m pytest         # 5 smoke tests

python3 scripts/run_train.py           # ~5 min GPU  (debug_mode: false in configs/)
python3 scripts/run_stress_test.py     # ~2 min GPU
python3 scripts/run_calibrate.py       # ~2 min GPU
python3 scripts/run_score.py          # ~2 min GPU
```

> Set `debug_mode: true` in `configs/default.yaml` to train on a 100k-sequence subset.

---

## Pipeline stages

| Script | What it does | Outputs |
|---|---|---|
| `run_train.py` | Train β-VAE; compute per-sequence test MSE | `outputs/train/` |
| `run_stress_test.py` | VAE-only detection across 8 corruption types | `outputs/stress_test/` |
| `run_calibrate.py` | Calibrate kinematic thresholds from clean quantiles; rerun stress tests | `outputs/calibrate/` |
| `run_score.py` | VAE + hand-picked kinematic scorer (baseline reference) | `outputs/score/` |

---

## Repository layout

```
VAESDE_model/
├── configs/default.yaml          experiment settings (epochs, lr, beta, seed, thresholds)
├── docs/results.md               extended results tables
├── scripts/                      entry-point scripts (one def main() each)
├── src/vaesde/                   installable package (pip install -e .)
│   ├── model.py                  SequenceVAE
│   ├── kinematics.py             compute_kinematics, compute_flags, calibrate_thresholds
│   ├── corruption.py             8 corruption functions
│   ├── inference.py              batched compute_recon_mse
│   ├── training.py               SequenceDataset, vae_loss, run_epoch
│   ├── normalisation.py
│   ├── io_utils.py
│   └── reporting/                per-stage CSV/NPZ save helpers
├── tests/test_smoke.py
├── data/                         input files (not tracked — copy from VAESDE_preprocess)
└── outputs/                      generated results (not tracked)
```

---

## Input data

Run VAESDE_preprocess first, then copy these files into `data/`:

```bash
PREPROCESS_REPO=/path/to/VAESDE_preprocess
cp $PREPROCESS_REPO/data/X_train.npy                 data/
cp $PREPROCESS_REPO/data/X_test.npy                  data/
cp $PREPROCESS_REPO/data/normalisation_mean.csv       data/
cp $PREPROCESS_REPO/data/normalisation_std.csv        data/
cp $PREPROCESS_REPO/data/train_sequence_metadata.csv  data/
cp $PREPROCESS_REPO/data/test_sequence_metadata.csv   data/
```

| File | Shape | Description |
|---|---|---|
| `X_train.npy` | (1 412 436, 30, 4) | Normalised train sequences |
| `X_test.npy` | (160 946, 30, 4) | Normalised test sequences |
| `normalisation_{mean,std}.csv` | 4 values each | Train-only per-feature stats |
| `{train,test}_sequence_metadata.csv` | — | `segment_id`, start/end time per sequence |

Features in order: `E_m`, `N_m`, `vE_mps`, `vN_mps`.

---

## Reproducibility

All randomness uses `np.random.default_rng(seed)` with `seed: 42` in `configs/default.yaml`.

```bash
python3 scripts/run_train.py
# Writes: outputs/train/sequence_vae_full.pt

# scripts 2–4 read from models/sequence_vae/:
cp outputs/train/sequence_vae_full.pt models/sequence_vae/sequence_vae_full.pt

python3 scripts/run_calibrate.py
# Writes: outputs/calibrate/calibrated_thresholds.csv
#         outputs/calibrate/calibrated_motion_prior_summary.csv
```

---

## Limitations

- The VAE alone does not detect position jumps below ~2000 m or stationary clutter; kinematic flags are required.
- Moderate speed scaling (1.5×–2×) and sudden turns remain only partially detected (~20–50% combined).
- All data is domain-specific ADS-B from fixed-wing GA aircraft; detection rates will differ on other aircraft types or sensors.
- Calibrated thresholds are data-derived and will shift with different traffic distributions.
