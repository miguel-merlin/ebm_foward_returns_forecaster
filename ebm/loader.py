import glob
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from typing import Dict


class DataLoader:
    """Load and preprocess ETF and macro data"""

    def __init__(self, etf_dir: str = "data/etfs", macro_dir: str = "data/macro"):
        self.etf_dir = Path(etf_dir)
        self.macro_dir = Path(macro_dir)

    def load_etf_data(self) -> pd.DataFrame:
        """Load all ETF data and combine"""
        etf_files = glob.glob(str(self.etf_dir / "*.csv"))

        dfs = []
        for file in etf_files:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            etf_name = Path(file).stem
            dfs.append(df[["Close"]].rename(columns={"Close": etf_name}))

        combined = pd.concat(dfs, axis=1)
        return combined.ffill()

    def load_macro_data(self) -> pd.DataFrame:
        """Load macro indicators"""
        macro_files = glob.glob(str(self.macro_dir / "*.csv"))

        dfs = []
        for file in macro_files:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            dfs.append(df)

        combined = pd.concat(dfs, axis=1)
        return combined.ffill()

    def prepare_training_data(
        self,
        etf_data: pd.DataFrame,
        macro_data: pd.DataFrame,
        lookback: int = 60,
        forward_horizon: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare training data with actual forward returns as targets

        Returns:
            Dictionary with past_prices, current_prices, macro_factors, actual_returns
        """
        etf_data = etf_data.copy()
        macro_data = macro_data.copy()

        # Coerce to DatetimeIndex, strip timezone, and normalize to daily granularity.
        etf_idx = pd.DatetimeIndex(pd.to_datetime(etf_data.index, errors="coerce"))
        macro_idx = pd.DatetimeIndex(pd.to_datetime(macro_data.index, errors="coerce"))
        if etf_idx.tz is not None:
            etf_idx = etf_idx.tz_localize(None)
        if macro_idx.tz is not None:
            macro_idx = macro_idx.tz_localize(None)
        etf_data.index = etf_idx.normalize()
        macro_data.index = macro_idx.normalize()

        common_dates = etf_data.index.intersection(macro_data.index)
        if len(common_dates) == 0:
            raise ValueError(
                "No overlapping dates between ETF and macro datasets after index alignment."
            )
        etf_aligned = etf_data.loc[common_dates]
        macro_aligned = macro_data.loc[common_dates]

        etf_returns = (
            etf_aligned.pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        macro_clean = macro_aligned.ffill().bfill()
        macro_std = macro_clean.std(ddof=0).replace(0, 1.0)
        macro_normalized = (macro_clean - macro_clean.mean()) / macro_std
        macro_normalized = (
            macro_normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

        etf_values = np.ascontiguousarray(etf_returns.to_numpy(dtype=np.float64))
        macro_values = np.ascontiguousarray(macro_normalized.to_numpy(dtype=np.float64))

        n_timesteps, n_assets = etf_values.shape
        n_samples_per_asset = n_timesteps - lookback - forward_horizon
        if n_samples_per_asset <= 0:
            raise ValueError(
                "Not enough overlapping data to build samples. "
                f"Need more than lookback + forward_horizon ({lookback + forward_horizon}) "
                f"timesteps, got {n_timesteps}."
            )

        # Build lookback windows for every asset at every valid timestep.
        # Shape: (n_samples_per_asset, n_assets, lookback)
        past_windows = np.lib.stride_tricks.sliding_window_view(
            etf_values, window_shape=lookback, axis=0
        )[:n_samples_per_asset]
        past_prices = np.ascontiguousarray(past_windows.reshape(-1, lookback))

        # Current returns at timestep i for each asset.
        # Shape before reshape: (n_samples_per_asset, n_assets)
        current_slice = etf_values[lookback : n_timesteps - forward_horizon]
        current_prices = np.ascontiguousarray(current_slice.reshape(-1, 1))

        # Macro factors at timestep i, repeated once per asset to preserve order:
        # i-major, then asset-major.
        macro_slice = macro_values[lookback : n_timesteps - forward_horizon]
        macro_factors = np.ascontiguousarray(np.repeat(macro_slice, n_assets, axis=0))

        # Forward return is sum of returns in [i, i + forward_horizon).
        cumsum = np.vstack([np.zeros((1, n_assets)), np.cumsum(etf_values, axis=0)])
        start = np.arange(lookback, n_timesteps - forward_horizon)
        forward_sum = cumsum[start + forward_horizon] - cumsum[start]
        actual_returns = np.ascontiguousarray(forward_sum.reshape(-1, 1))

        return {
            "past_prices": torch.FloatTensor(past_prices),
            "current_prices": torch.FloatTensor(current_prices),
            "macro_factors": torch.FloatTensor(macro_factors),
            "actual_returns": torch.FloatTensor(actual_returns),
        }
