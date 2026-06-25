"""Shared I/O helpers: config loading and normalisation stats."""

import numpy as np
import pandas as pd
import yaml
from vaesde.constants import FEATURES


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_norm_stats(mean_path: str, std_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load normalisation mean and std CSVs; return arrays of shape (4,)."""
    mean_df = pd.read_csv(mean_path, index_col=0)
    std_df  = pd.read_csv(std_path,  index_col=0)
    for df, tag in [(mean_df, "mean"), (std_df, "std")]:
        missing = [f for f in FEATURES if f not in df.index]
        assert not missing, f"normalisation_{tag}.csv missing features: {missing}"
    mean = mean_df.loc[FEATURES, "mean"].values.astype(np.float64)
    std  = std_df.loc[FEATURES,  "std"].values.astype(np.float64)
    return mean, std
