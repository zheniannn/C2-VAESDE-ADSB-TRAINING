# Extended results

## VAE reconstruction MSE — full model, 160 946 test sequences

| Percentile | MSE |
|---|---:|
| p50 | 0.00325 |
| p90 | 0.02144 |
| p95 | **0.03675** ← anomaly threshold |
| p99 | **0.09542** ← strict threshold |

Both thresholds are stored in `configs/default.yaml` under `vae_thresholds`.

---

## VAE-only stress test — 50 000 sequences, seed 42

| Corruption | Mean MSE | % above p95 |
|---|---:|---:|
| clean (baseline) | 0.00915 | 4.5% |
| speed_scaled_1.5× | 0.02061 | 12.9% |
| speed_scaled_2.0× | 0.03857 | 24.2% |
| position_jump_1000 m | 0.00919 | 4.5% |
| position_jump_2000 m | 0.00928 | 4.5% |
| random_walk_velocity | 0.12564 | **100%** |
| sudden_turn_90° | 0.02070 | 9.2% |
| stationary_clutter | 0.00059 | 0% |

The VAE is a velocity-distribution anomaly detector. Position jumps and stationary clutter produce no shift in normalised velocity patterns, so reconstruction error is indistinguishable from clean sequences.

---

## Calibrated kinematic thresholds — derived at p99.5 from clean test sequences

| Threshold | Value | Source |
|---|---:|---|
| max_speed | 150.0 m/s | physical floor (≥ p99.5) |
| max_accel | 5.77 m/s² | clean p99.5 |
| mean_pv_error | 224.8 m | clean p99.5 |
| max_pv_error | 1955.4 m | clean p99.5 |
| mean_turn_rate | 4.37 °/s | clean p99.5 |
| max_turn_rate | 15.14 °/s | clean p99.5 |
| min_mean_speed | 1.0 m/s | lower floor (≤ p0.5) |
| min_displacement | 50.0 m | lower floor (≤ p0.5) |

Calibration reduced the clean false-flag rate from 20.6 % → 8.9 %.

---

## Full combined detection table

| Corruption | VAE p95 | Kinematic | Combined p95 |
|---|---:|---:|---:|
| clean (false-flag rate) | 4.5% | 5.5% | **8.9%** |
| speed_scaled_1.5× | 12.9% | 14.5% | 20.9% |
| speed_scaled_2.0× | 24.3% | 44.1% | 48.6% |
| position_jump_1000 m | 4.5% | 5.8% | 8.9% |
| position_jump_2000 m | 4.5% | 92.5% | 93.0% |
| random_walk_velocity | 100% | 47.4% | 100% |
| sudden_turn_90° | 9.1% | 13.2% | 19.5% |
| stationary_clutter | 0% | 100% | 100% |

The 1000 m position jump is not caught: the calibrated max position-velocity error threshold (≈ 1955 m) exceeds the jump magnitude, so neither the VAE nor the kinematic flag fires.
