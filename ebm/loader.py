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
        return combined.fillna(method="ffill")

    def load_macro_data(self) -> pd.DataFrame:
        """Load macro indicators"""
        macro_files = glob.glob(str(self.macro_dir / "*.csv"))

        dfs = []
        for file in macro_files:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            dfs.append(df)

        combined = pd.concat(dfs, axis=1)
        return combined.fillna(method="ffill")

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
        common_dates = etf_data.index.intersection(macro_data.index)
        etf_aligned = etf_data.loc[common_dates]
        macro_aligned = macro_data.loc[common_dates]

        etf_returns = etf_aligned.pct_change().fillna(0)
        macro_normalized = (macro_aligned - macro_aligned.mean()) / macro_aligned.std()

        past_prices_list = []
        current_prices_list = []
        macro_list = []
        actual_returns_list = []

        for i in range(lookback, len(etf_aligned) - forward_horizon):
            for col in etf_returns.columns:
                past = etf_returns[col].iloc[i - lookback : i].values
                current = np.array([etf_returns[col].iloc[i]])
                macro = macro_normalized.iloc[i].values
                actual_return = etf_returns[col].iloc[i : i + forward_horizon].sum()

                past_prices_list.append(past)
                current_prices_list.append(current)
                macro_list.append(macro)
                actual_returns_list.append(actual_return)

        return {
            "past_prices": torch.FloatTensor(np.array(past_prices_list)),
            "current_prices": torch.FloatTensor(np.array(current_prices_list)),
            "macro_factors": torch.FloatTensor(np.array(macro_list)),
            "actual_returns": torch.FloatTensor(
                np.array(actual_returns_list)
            ).unsqueeze(1),
        }
