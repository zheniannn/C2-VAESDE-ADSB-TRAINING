"""Trajectory corruption functions for stress-testing the VAE.

All functions accept physical-unit arrays (N, 30, 4) and return physical-unit arrays.
"""

import numpy as np
from vaesde.constants import IDX_E, IDX_N, IDX_VE, IDX_VN, DT


def _integrate(s: np.ndarray) -> np.ndarray:
    """Rebuild E_m and N_m from velocities, keeping position[0] fixed."""
    s[:, 1:, IDX_E] = s[:, 0:1, IDX_E] + np.cumsum(s[:, :-1, IDX_VE], axis=1) * DT
    s[:, 1:, IDX_N] = s[:, 0:1, IDX_N] + np.cumsum(s[:, :-1, IDX_VN], axis=1) * DT
    return s


def corrupt_speed_scale(p: np.ndarray, factor: float) -> np.ndarray:
    """Scale vE and vN by factor, then reintegrate positions."""
    s = p.copy()
    s[:, :, IDX_VE] *= factor
    s[:, :, IDX_VN] *= factor
    return _integrate(s)


def corrupt_position_jump(p: np.ndarray, jump_m: float) -> np.ndarray:
    """Add discontinuous +jump_m to E_m from timestep 15 onward; velocities unchanged."""
    s = p.copy()
    s[:, 15:, IDX_E] += jump_m
    return s


def corrupt_random_walk_velocity(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise (std=20 m/s) to velocities, then reintegrate positions."""
    s = p.copy()
    n, seq_len = s.shape[0], s.shape[1]
    s[:, :, IDX_VE] += rng.normal(0.0, 20.0, (n, seq_len))
    s[:, :, IDX_VN] += rng.normal(0.0, 20.0, (n, seq_len))
    return _integrate(s)


def corrupt_sudden_turn(p: np.ndarray) -> np.ndarray:
    """Rotate velocity vector 90° from timestep 15 onward, then reintegrate."""
    s = p.copy()
    vE_orig = s[:, :, IDX_VE].copy()
    vN_orig = s[:, :, IDX_VN].copy()
    s[:, 15:, IDX_VE] = -vN_orig[:, 15:]
    s[:, 15:, IDX_VN] =  vE_orig[:, 15:]
    return _integrate(s)


def corrupt_stationary(p: np.ndarray) -> np.ndarray:
    """Fix position to timestep-0 values; set all velocities to zero."""
    s = p.copy()
    s[:, :, IDX_E]  = s[:, 0:1, IDX_E]
    s[:, :, IDX_N]  = s[:, 0:1, IDX_N]
    s[:, :, IDX_VE] = 0.0
    s[:, :, IDX_VN] = 0.0
    return s
