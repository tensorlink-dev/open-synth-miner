"""Market data loader utilities with Hydra-friendly instantiation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch


@dataclass
class MarketDataLoader:
    """Loads market data slices and provides backtest windows.

    This class is intentionally lightweight and Hydra-instantiable. It supports
    pluggable symbol/timeframe selections and can deliver deterministic slices
    for champion-vs-challenger evaluations.
    """

    symbols: List[str]
    timeframe: str
    window_size: int
    feature_cols: List[str] = field(default_factory=list)
    data_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.frame = self._load_frame()

    def _load_frame(self) -> pd.DataFrame:
        if self.data_path:
            df = pd.read_csv(self.data_path)
        else:
            # Fallback synthetic frame for development.
            steps = self.window_size * 10
            idx = pd.date_range("2024-01-01", periods=steps, freq=self.timeframe)
            data = {symbol: 100 + np.cumsum(np.random.randn(steps)) for symbol in self.symbols}
            df = pd.DataFrame(data, index=idx)
        return df

    def _slice(self, start: int, end: int) -> torch.Tensor:
        sliced = self.frame.iloc[start:end]
        return torch.tensor(sliced.values, dtype=torch.float32)

    def get_backtest_window(self, start: int, end: int) -> dict:
        """Return a dict containing prices and optional features for backtesting."""
        prices = self._slice(start, end)
        covariates = None
        if self.feature_cols:
            feature_slice = self.frame[self.feature_cols].iloc[start:end]
            covariates = torch.tensor(feature_slice.values, dtype=torch.float32)
        return {"prices": prices, "covariates": covariates}

    def latest_window(self) -> dict:
        end = len(self.frame)
        start = max(0, end - self.window_size)
        return self.get_backtest_window(start, end)


__all__ = ["MarketDataLoader"]
