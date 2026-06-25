"""Normalisation helpers for ENU trajectory arrays."""

import numpy as np


def denormalise(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert normalised array back to physical units."""
    return arr * std + mean


def renormalise(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert physical-unit array to normalised form."""
    return (arr - mean) / std
