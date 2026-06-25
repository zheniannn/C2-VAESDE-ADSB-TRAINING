# VAESDE_model — GA Trajectory Motion Prior (Modelling)

Sequence VAE and motion-prior scorer for fixed-wing GA aircraft trajectories. Consumes normalised sequence arrays produced by the **VAESDE_preprocess** repository.

---

## Setup: copy input files from VAESDE_preprocess

Run the VAESDE_preprocess pipeline first (all 8 steps), then copy these files into `data/`:

```bash
PREPROCESS_REPO=/path/to/VAESDE_preprocess

cp $PREPROCESS_REPO/data/X_train.npy                 data/
cp $PREPROCESS_REPO/data/X_test.npy                  data/
cp $PREPROCESS_REPO/data/normalisation_mean.csv       data/
cp $PREPROCESS_REPO/data/normalisation_std.csv        data/
cp $PREPROCESS_REPO/data/train_sequence_metadata.csv  data/
cp $PREPROCESS_REPO/data/test_sequence_metadata.csv   data/
```

| File | Shape / Size | Description |
|---|---|---|
| `X_train.npy` | (1 412 436, 30, 4) / 1.36 GB | Normalised train sequences |
| `X_test.npy` | (160 946, 30, 4) / 155 MB | Normalised test sequences |
| `normalisation_mean.csv` | 4 values | Train-only per-feature means |
| `normalisation_std.csv` | 4 values | Train-only per-feature stds |
| `train_sequence_metadata.csv` | 1 412 436 rows | segment_id / start / end time per train sequence |
| `test_sequence_metadata.csv` | 160 946 rows | segment_id / start / end time per test sequence |

Features (in order): `E_m`, `N_m`, `vE_mps`, `vN_mps` — horizontal ENU position and velocity, normalised to zero mean / unit std on the train split.

---

## Running the pipeline

```bash
python3 stage_01_train_sequence_vae.py                   # ~5 min on GPU  (set DEBUG_MODE=False)
python3 stage_02_stress_test_sequence_vae.py             # ~2 min on GPU
python3 stage_03_calibrate_motion_prior_thresholds.py    # ~2 min on GPU
python3 stage_04_motion_prior_scorer.py                  # ~2 min on GPU
```

---

## Repository structure

```
VAESDE_model/
├── stage_01_train_sequence_vae.py                # Train sequence VAE (debug + full mode)
├── stage_02_stress_test_sequence_vae.py          # VAE-only stress test across 8 corruption types
├── stage_03_calibrate_motion_prior_thresholds.py # Calibrate thresholds from clean quantiles
├── stage_04_motion_prior_scorer.py               # VAE + hand-picked kinematic scorer
│
├── data/                                         # Input files (not tracked — copy from VAESDE_preprocess)
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── normalisation_mean.csv
│   ├── normalisation_std.csv
│   ├── train_sequence_metadata.csv
│   └── test_sequence_metadata.csv
│
└── models/                                       # Generated outputs (gitignored)
    └── sequence_vae/
        ├── sequence_vae_full.pt
        ├── loss_history.csv
        ├── reconstruction_error_summary.csv
        ├── stress_tests/
        │   ├── stress_test_summary.csv
        │   └── stress_test_per_sequence_errors.csv
        └── motion_prior_scorer/
            ├── motion_prior_summary.csv
            └── calibrated/
                ├── calibrated_thresholds.csv
                └── calibrated_motion_prior_summary.csv
```

---

## Dependencies

```bash
pip install torch pandas numpy
```

Python 3.10+ required.

---

## Stage details

### stage_01 — Sequence VAE

Trains a β-VAE on the 30-step sliding-window sequences.

**Architecture**

```
Encoder: Linear(120→256)→ReLU→Linear(256→128)→ReLU → μ, log σ²  (latent dim 16)
Decoder: Linear(16→128)→ReLU→Linear(128→256)→ReLU→Linear(256→120)
```

- Loss: `MSE_recon + 0.001 × KL`
- Full training: 30 epochs over 1 412 436 sequences
- Set `DEBUG_MODE = False` at the top of the script for full training

**Test reconstruction MSE (full model)**

| Percentile | Value |
|-----------|-------|
| p50 | 0.003248 |
| p90 | 0.021440 |
| p95 | **0.036746** ← VAE anomaly threshold |
| p99 | **0.095424** ← stricter threshold |

---

### stage_02 — VAE-only stress test

Evaluates reconstruction MSE against 8 physically motivated corruption types (n = 50 000 sequences, seed 42).

| Corruption | VAE p95 flag | Interpretation |
|---|---|---|
| clean | ~5% | calibration baseline |
| speed_scaled_1.5× | ~13% | partially detected |
| speed_scaled_2.0× | ~24% | partially detected |
| position_jump_1000 m | ~5% | **not detected** — VAE is velocity-distribution aware, not position aware |
| position_jump_2000 m | ~5% | not detected |
| random_walk_velocity | **100%** | strongly detected |
| sudden_turn_90° | ~9% | partially detected |
| stationary_clutter | 0% | **not detected** |

**Key insight:** the VAE alone is a velocity-distribution anomaly detector. Position jumps and stationary clutter require kinematic consistency checks.

---

### stage_03 — Calibrate motion prior thresholds

Derives all kinematic thresholds from clean test-set quantiles (p99.5 upper-tail, p0.5 lower-tail) and adds a turn-rate feature for better sudden-manoeuvre detection.

**Calibrated thresholds**

| Threshold | Value | Source |
|---|---|---|
| max_speed | 150.0 m/s | physical floor |
| max_accel | 5.77 m/s² | clean p99.5 |
| mean_pv_error | 224.8 m | clean p99.5 |
| max_pv_error | 1955.4 m | clean p99.5 |
| mean_turn_rate | 4.37 °/s | clean p99.5 |
| max_turn_rate | 15.14 °/s | clean p99.5 |
| min_mean_speed | 1.0 m/s | lower floor |
| min_displacement | 50.0 m | lower floor |

Calibration reduced the clean false-flag rate from **20.6% → 8.9%**.

---

### stage_04 — Motion prior scorer (hand-picked thresholds)

Combines VAE reconstruction MSE with six hand-picked kinematic flags. Provided as a reference baseline; use stage_03 outputs for calibrated thresholds.

**Combined detection rates**

| Test type | VAE p95 | Kinematic | Combined p95 |
|---|---|---|---|
| clean | 4.5% | 18.7% | 20.3% |
| speed_scaled_2.0× | 24.3% | 30.7% | 45.3% |
| position_jump_1000 m | 4.5% | 100.0% | 100.0% |
| position_jump_2000 m | 4.5% | 100.0% | 100.0% |
| random_walk_velocity | 100.0% | 1.7% | 100.0% |
| stationary_clutter | 0.0% | 100.0% | 100.0% |
