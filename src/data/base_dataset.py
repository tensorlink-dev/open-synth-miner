"""Base dataset and feature engineering utilities for time series backtesting."""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
from torch.utils.data import Dataset


class FeatureEngineerBase:
    """Base feature engineering hook for time series datasets.

    Subclass this to add domain-specific transforms. All methods are expected to be
    side-effect free and return tensors compatible with model consumption.
    """

    def transform_history(self, history: torch.Tensor) -> torch.Tensor:
        """Transform the input history window (e.g., normalization)."""
        return history

    def transform_target(self, target: torch.Tensor) -> torch.Tensor:
        """Transform the target series if needed."""
        return target

    def extra_features(self, history: torch.Tensor) -> Optional[torch.Tensor]:
        """Optionally return additional features derived from the history window."""
        return None


class StridedTimeSeriesDataset(Dataset):
    """Dataset that yields strided context/target pairs with optional covariates.

    The dataset builds sliding windows from a target series using ``context_len`` and
    ``pred_len`` with a configurable ``stride``. It can also return aligned
    past/future covariates to support richer hybrid architectures.
    """

    def __init__(
        self,
        target: torch.Tensor,
        context_len: int,
        pred_len: int,
        stride: int = 1,
        *,
        past_covariates: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None,
        feature_engineer: Optional[FeatureEngineerBase] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        if target.dim() != 2:
            raise ValueError("target tensor must be 1D or 2D time series")
        if context_len <= 0 or pred_len <= 0:
            raise ValueError("context_len and pred_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.target = target.float()
        self.context_len = context_len
        self.pred_len = pred_len
        self.stride = stride
        self.feature_engineer = feature_engineer or FeatureEngineerBase()
        self.target_transform = target_transform

        if past_covariates is not None:
            self._validate_covariates(past_covariates, "past_covariates")
            self.past_covariates = past_covariates.float()
        else:
            self.past_covariates = None

        if future_covariates is not None:
            self._validate_covariates(future_covariates, "future_covariates")
            self.future_covariates = future_covariates.float()
        else:
            self.future_covariates = None

        self.indices = self._build_indices()

    def _validate_covariates(self, covariates: torch.Tensor, name: str) -> None:
        if covariates.dim() == 1:
            covariates = covariates.unsqueeze(-1)
        if covariates.dim() != 2:
            raise ValueError(f"{name} must be 1D or 2D with shape [T, F]")
        if covariates.shape[0] < self.target.shape[0]:
            raise ValueError(
                f"{name} length {covariates.shape[0]} shorter than target length {self.target.shape[0]}"
            )

    def _build_indices(self) -> List[int]:
        stops = self.target.shape[0] - (self.context_len + self.pred_len) + 1
        if stops < 1:
            raise ValueError("series is too short for the requested context_len and pred_len")
        return list(range(0, stops, self.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        start = self.indices[idx]
        hist_end = start + self.context_len
        target_end = hist_end + self.pred_len

        history = self.target[start:hist_end]
        target_series = self.target[hist_end:target_end]

        history = self.feature_engineer.transform_history(history)
        target_series = self.feature_engineer.transform_target(target_series)
        engineered = self.feature_engineer.extra_features(history)

        if self.target_transform:
            target_series = self.target_transform(target_series)

        sample = {
            "history": history.squeeze(-1),
            "target": target_series[-1].squeeze(-1),  # final step target for convenience
            "initial_price": history[-1].squeeze(-1),
            "actual_series": target_series.squeeze(-1),
        }

        if self.past_covariates is not None:
            sample["past_covariates"] = self.past_covariates[start:hist_end]
        if self.future_covariates is not None:
            # Provide context+future slice so models can fuse both parts
            sample["future_covariates"] = self.future_covariates[start:target_end]
            sample["future_covariates_pred"] = self.future_covariates[hist_end:target_end]

        if engineered is not None:
            sample["engineered_features"] = engineered
        return sample
