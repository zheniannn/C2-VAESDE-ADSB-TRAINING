"""Kinematic feature extraction and flag computation for trajectory sequences.

Uses the stage-03 feature set (9 metrics including turn-rate).
"""

import numpy as np
from vaesde.constants import IDX_E, IDX_N, IDX_VE, IDX_VN, DT, SPEED_VALID_THRESH


def compute_kinematics(seqs_phys: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-sequence kinematic scalars (each shape (N,)) from physical-unit sequences."""
    E  = seqs_phys[:, :, IDX_E]
    N_ = seqs_phys[:, :, IDX_N]
    vE = seqs_phys[:, :, IDX_VE]
    vN = seqs_phys[:, :, IDX_VN]

    speed = np.sqrt(vE**2 + vN**2)
    aE    = np.gradient(vE, DT, axis=1)
    aN    = np.gradient(vN, DT, axis=1)
    accel = np.sqrt(aE**2 + aN**2)

    pv_err = np.sqrt(
        (E[:, 1:] - (E[:, :-1] + vE[:, :-1] * DT))**2 +
        (N_[:, 1:] - (N_[:, :-1] + vN[:, :-1] * DT))**2
    )

    heading    = np.arctan2(vN, vE)
    heading_uw = np.unwrap(heading, axis=1)
    hd_change  = np.diff(heading_uw, axis=1)
    tr         = np.abs(np.degrees(hd_change)) / DT   # deg/s

    valid       = (speed[:, :-1] >= SPEED_VALID_THRESH) & (speed[:, 1:] >= SPEED_VALID_THRESH)
    valid_count = valid.sum(axis=1)
    tr_v        = np.where(valid, tr, 0.0)

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


def calibrate_thresholds(clean_kin: dict[str, np.ndarray]) -> tuple[dict, "pd.DataFrame"]:
    """Derive per-flag thresholds from clean-sequence quantiles at the 0.5% false-flag rate."""
    import pandas as pd
    upper_metrics = ["max_speed_mps", "max_accel_mps2", "mean_pv_error_m",
                     "max_pv_error_m", "mean_turn_rate_degps", "max_turn_rate_degps"]
    lower_metrics = ["mean_speed_mps", "total_displacement_m"]
    upper_qs      = [90, 95, 99, 99.5, 99.9]
    lower_qs      = [0.1, 0.5, 1.0, 2.0, 5.0]

    rows: list[dict] = []
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
    return thresholds, pd.DataFrame(rows)


def compute_flags(kin: dict[str, np.ndarray],
                  recon_mse: np.ndarray,
                  thr: dict) -> dict[str, np.ndarray]:
    """Apply threshold dict to kinematic features and VAE MSE.

    thr keys used:
      vae_p95_threshold, vae_p99_threshold
      max_speed_threshold_mps, max_accel_threshold_mps2
      mean_pv_error_threshold_m, max_pv_error_threshold_m
      min_mean_speed_threshold_mps, min_displacement_threshold_m
      mean_turn_rate_threshold_degps, max_turn_rate_threshold_degps  (optional)
    """
    fs  = kin["max_speed_mps"]        > thr["max_speed_threshold_mps"]
    fa  = kin["max_accel_mps2"]       > thr["max_accel_threshold_mps2"]
    fpx = kin["max_pv_error_m"]       > thr["max_pv_error_threshold_m"]
    fpm = kin["mean_pv_error_m"]      > thr["mean_pv_error_threshold_m"]
    fts = kin["mean_speed_mps"]       < thr["min_mean_speed_threshold_mps"]
    fld = kin["total_displacement_m"] < thr["min_displacement_threshold_m"]

    n = len(recon_mse)
    flag_turn_max  = np.zeros(n, dtype=bool)
    flag_turn_mean = np.zeros(n, dtype=bool)
    if "max_turn_rate_threshold_degps" in thr:
        flag_turn_max  = kin["max_turn_rate_degps"]  > thr["max_turn_rate_threshold_degps"]
        flag_turn_mean = kin["mean_turn_rate_degps"] > thr["mean_turn_rate_threshold_degps"]

    kflag = fs | fa | fpx | fpm | fts | fld | flag_turn_max | flag_turn_mean

    vp95 = recon_mse > thr["vae_p95_threshold"]
    vp99 = recon_mse > thr["vae_p99_threshold"]

    return {
        "vae_flag_p95":          vp95,
        "vae_flag_p99":          vp99,
        "flag_speed":            fs,
        "flag_accel":            fa,
        "flag_pv_max":           fpx,
        "flag_pv_mean":          fpm,
        "flag_too_slow":         fts,
        "flag_low_displacement": fld,
        "flag_turn_max":         flag_turn_max,
        "flag_turn_mean":        flag_turn_mean,
        "kinematic_flag":        kflag,
        "combined_flag_p95":     vp95 | kflag,
        "combined_flag_p99":     vp99 | kflag,
    }
