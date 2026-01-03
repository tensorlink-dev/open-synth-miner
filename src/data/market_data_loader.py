"""Leak-safe market data loader with modular feature engineering."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pywt
import torch
from huggingface_hub import hf_hub_download
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset


# ---------------------------------------------------------------------------
# Feature engineering strategies
# ---------------------------------------------------------------------------


class FeatureEngineer(abc.ABC):
    """Abstract strategy for transforming raw prices into model tensors."""

    @abc.abstractmethod
    def prepare_cache(self, prices: np.ndarray) -> Any:
        """Pre-compute causal, leakage-safe artifacts for a full series."""

    @abc.abstractmethod
    def make_input(self, cache: Any, start: int, length: int) -> torch.Tensor:
        """Create the model input tensor for a window starting at ``start``."""

    @abc.abstractmethod
    def make_target(self, cache: Any, start: int, length: int) -> torch.Tensor:
        """Create the model target tensor for a window starting at ``start``."""

    @abc.abstractmethod
    def get_volatility(self, cache: Any, start: int, length: int) -> float:
        """Return a scalar volatility proxy for stratification buckets."""

    def clean_prices(self, prices: np.ndarray) -> np.ndarray:
        """Standard causal cleaning to avoid NaNs or non-positive values."""

        p = np.asarray(prices, dtype=np.float64)
        p[~np.isfinite(p)] = np.nan
        series = pd.Series(p).ffill()
        first_valid = 1.0 if series.dropna().empty else float(series.dropna().iloc[0])
        series = series.fillna(first_valid)
        p = series.values
        p = np.clip(p, 1e-6, None)
        return p


class ZScoreEngineer(FeatureEngineer):
    """Rolling z-score features backed by cached computations."""

    def __init__(self, short_win: int = 20, long_win: int = 200) -> None:
        self.short_win = short_win
        self.long_win = long_win

    def prepare_cache(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        p = self.clean_prices(prices)
        log_prices = np.log(p + 1e-12)

        returns = np.diff(log_prices, prepend=log_prices[0]).astype(np.float32)
        returns[~np.isfinite(returns)] = 0.0

        series = pd.Series(log_prices)
        z_short = (series - series.rolling(self.short_win).mean()) / (
            series.rolling(self.short_win).std() + 1e-8
        )
        z_long = (series - series.rolling(self.long_win).mean()) / (
            series.rolling(self.long_win).std() + 1e-8
        )

        features = np.stack(
            [
                returns.astype(np.float32),
                z_short.fillna(0.0).values.astype(np.float32),
                z_long.fillna(0.0).values.astype(np.float32),
            ],
            axis=1,
        )
        return {"features": features, "returns": returns}

    def make_input(self, cache: Any, start: int, length: int) -> torch.Tensor:
        window = cache["features"][start : start + length]
        return torch.from_numpy(window).float().T

    def make_target(self, cache: Any, start: int, length: int) -> torch.Tensor:
        target = cache["features"][start : start + length, 0:1]
        return torch.from_numpy(target).float().T

    def get_volatility(self, cache: Any, start: int, length: int) -> float:
        window = cache["returns"][start : start + length]
        return float(np.std(window))


class WaveletEngineer(FeatureEngineer):
    """On-the-fly wavelet decomposition with cached returns."""

    def __init__(self, wavelet: str = "db4", level: int = 4) -> None:
        self.wavelet = wavelet
        self.level = level

    def prepare_cache(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        p = self.clean_prices(prices)
        log_prices = np.log(p + 1e-12)
        returns = np.diff(log_prices, prepend=log_prices[0]).astype(np.float32)
        returns[~np.isfinite(returns)] = 0.0
        return {"returns": returns}

    def make_input(self, cache: Any, start: int, length: int) -> torch.Tensor:
        returns = cache["returns"][start : start + length].astype(np.float64)
        coeffs = pywt.wavedec(returns, wavelet=self.wavelet, level=self.level, mode="periodization")
        total_len = returns.size

        def reconstruct_single(idx: int) -> np.ndarray:
            keep = [np.zeros_like(c) for c in coeffs]
            if idx < len(coeffs):
                keep[idx] = coeffs[idx]
            reconstructed = pywt.waverec(keep, wavelet=self.wavelet, mode="periodization")
            return reconstructed[:total_len]

        approx = reconstruct_single(0)
        detail_L = reconstruct_single(1) if len(coeffs) > 1 else np.zeros(total_len)
        detail_Lm1 = reconstruct_single(2) if len(coeffs) > 2 else np.zeros(total_len)
        detail_Lm2 = reconstruct_single(3) if len(coeffs) > 3 else np.zeros(total_len)

        std = float(np.std(returns)) + 1e-12
        stack: List[np.ndarray] = []
        for arr in [returns, approx, detail_L, detail_Lm1, detail_Lm2]:
            clipped = np.clip(arr, -5.0 * std, 5.0 * std)
            stack.append(clipped.astype(np.float32))

        return torch.from_numpy(np.stack(stack, axis=1)).float().T

    def make_target(self, cache: Any, start: int, length: int) -> torch.Tensor:
        target = cache["returns"][start : start + length]
        return torch.from_numpy(target[None, :]).float()

    def get_volatility(self, cache: Any, start: int, length: int) -> float:
        window = cache["returns"][start : start + length]
        return float(np.std(window))


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


@dataclass
class AssetData:
    name: str
    timestamps: np.ndarray
    prices: np.ndarray


class DataSource(abc.ABC):
    """Abstract data source. Implementations return per-asset series."""

    @abc.abstractmethod
    def load_data(self, assets: List[str]) -> List[AssetData]:
        """Load asset data for the requested tickers."""


class HFParquetSource(DataSource):
    """Loads parquet data from Hugging Face Hub."""

    def __init__(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: Optional[str] = None,
        repo_type: Optional[str] = "dataset",
        asset_column: str = "asset",
        price_column: str = "price",
        timestamp_column: str = "timestamp",
    ) -> None:
        self.repo_id = repo_id
        self.filename = filename
        self.revision = revision
        self.repo_type = repo_type
        self.asset_column = asset_column
        self.price_column = price_column
        self.timestamp_column = timestamp_column

    def load_data(self, assets: List[str]) -> List[AssetData]:
        file_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            revision=self.revision,
            repo_type=self.repo_type,
        )
        table = pq.read_table(file_path)
        frame = table.to_pandas()

        if self.timestamp_column not in frame.columns:
            raise ValueError("Parquet source must include a timestamp column")
        if self.price_column not in frame.columns:
            raise ValueError("Parquet source must include a price column")

        frame[self.timestamp_column] = pd.to_datetime(frame[self.timestamp_column], utc=True)

        results: List[AssetData] = []
        if self.asset_column in frame.columns:
            for asset in assets:
                subset = frame.loc[frame[self.asset_column] == asset]
                if subset.empty:
                    continue
                subset = subset.sort_values(self.timestamp_column)
                results.append(
                    AssetData(
                        name=asset,
                        timestamps=subset[self.timestamp_column].to_numpy(),
                        prices=subset[self.price_column].to_numpy(),
                    )
                )
        else:
            if len(assets) > 1:
                raise ValueError("Parquet source missing asset column; only one asset is supported")
            subset = frame.sort_values(self.timestamp_column)
            asset = assets[0] if assets else "asset"
            results.append(
                AssetData(
                    name=asset,
                    timestamps=subset[self.timestamp_column].to_numpy(),
                    prices=subset[self.price_column].to_numpy(),
                )
            )
        return results


class MockDataSource(DataSource):
    """Synthetic random-walk generator for testing and documentation."""

    def __init__(self, length: int = 5000, freq: str = "5min", seed: int = 7, base_price: float = 100.0) -> None:
        self.length = length
        self.freq = freq
        self.seed = seed
        self.base_price = base_price

    def load_data(self, assets: List[str]) -> List[AssetData]:
        rng = np.random.default_rng(self.seed)
        timestamps = pd.date_range("2020-01-01", periods=self.length, freq=self.freq, tz="UTC").to_numpy()
        results: List[AssetData] = []
        for idx, asset in enumerate(assets):
            walk = rng.standard_normal(self.length) * 0.001
            prices = self.base_price * np.exp(np.cumsum(walk)) * (1.0 + idx * 0.01)
            results.append(AssetData(name=asset, timestamps=timestamps.copy(), prices=prices))
        return results


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class WindowIndex:
    asset_idx: int
    start: int
    start_ts: pd.Timestamp
    decision_ts: pd.Timestamp
    horizon_ts: pd.Timestamp
    vol_bucket: int


class MarketDataset(Dataset):
    """Dataset that defers feature generation to a :class:`FeatureEngineer`."""

    def __init__(
        self,
        assets_data: List[AssetData],
        engineer: FeatureEngineer,
        input_len: int,
        pred_len: int,
        stride: Optional[int] = None,
        vol_thresholds: Tuple[float, float] = (0.0006, 0.0018),
    ) -> None:
        self.assets = assets_data
        self.engineer = engineer
        self.input_len = input_len
        self.pred_len = pred_len
        self.stride = stride or (input_len + pred_len)
        self.vol_thresholds = vol_thresholds

        self.caches: List[Any] = []
        self.indices: List[WindowIndex] = []
        self._build_indices()

    def _build_indices(self) -> None:
        def _ts(value: Any) -> pd.Timestamp:
            return pd.to_datetime(value, utc=True)

        for asset_idx, asset in enumerate(self.assets):
            cache = self.engineer.prepare_cache(asset.prices)
            self.caches.append(cache)

            total = len(asset.timestamps)
            max_start = total - (self.input_len + self.pred_len)
            if max_start < 0:
                continue

            for start in range(0, max_start + 1, self.stride):
                decision_pos = start + self.input_len - 1
                horizon_pos = decision_pos + self.pred_len
                start_ts = _ts(asset.timestamps[start])
                decision_ts = _ts(asset.timestamps[decision_pos])
                horizon_ts = _ts(asset.timestamps[horizon_pos])

                vol = self.engineer.get_volatility(cache, start, self.input_len)
                if vol < self.vol_thresholds[0]:
                    vol_bucket = 0
                elif vol < self.vol_thresholds[1]:
                    vol_bucket = 1
                else:
                    vol_bucket = 2

                self.indices.append(
                    WindowIndex(
                        asset_idx=asset_idx,
                        start=start,
                        start_ts=start_ts,
                        decision_ts=decision_ts,
                        horizon_ts=horizon_ts,
                        vol_bucket=vol_bucket,
                    )
                )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        window = self.indices[idx]
        cache = self.caches[window.asset_idx]

        inputs = self.engineer.make_input(cache, window.start, self.input_len)
        target = self.engineer.make_target(cache, window.start + self.input_len, self.pred_len)

        meta = {
            "asset_idx": window.asset_idx,
            "vol_bucket": window.vol_bucket,
            "start_index": window.start,
        }
        return {
            "inputs": inputs,
            "target": target,
            "decision_timestamp": window.decision_ts,
            "meta": meta,
        }

    def get_start_timestamps(self) -> np.ndarray:
        return np.array([w.start_ts for w in self.indices])

    def get_decision_timestamps(self) -> np.ndarray:
        return np.array([w.decision_ts for w in self.indices])

    def get_horizon_timestamps(self) -> np.ndarray:
        return np.array([w.horizon_ts for w in self.indices])

    def get_vol_buckets(self) -> np.ndarray:
        return np.array([w.vol_bucket for w in self.indices], dtype=int)


# ---------------------------------------------------------------------------
# Data loader with validation strategies
# ---------------------------------------------------------------------------


@dataclass
class MarketDataLoader:
    data_source: DataSource
    engineer: FeatureEngineer
    assets: List[str]
    input_len: int
    pred_len: int
    batch_size: int = 64

    def __post_init__(self) -> None:
        self.assets_data = self.data_source.load_data(self.assets)
        if not self.assets_data:
            raise ValueError("Data source returned no assets")
        stride = self.input_len + self.pred_len
        self.dataset = MarketDataset(
            self.assets_data,
            engineer=self.engineer,
            input_len=self.input_len,
            pred_len=self.pred_len,
            stride=stride,
        )

    # ----------------------------
    # Utility helpers
    # ----------------------------
    def _resolve_indices(self, ds: Dataset) -> Tuple[MarketDataset, np.ndarray]:
        if isinstance(ds, Subset):
            base_ds, parent_indices = self._resolve_indices(ds.dataset)
            return base_ds, np.asarray(parent_indices)[np.asarray(ds.indices)]
        if isinstance(ds, MarketDataset):
            return ds, np.arange(len(ds))
        raise TypeError("Unsupported dataset type for temporal assertions")

    def _build_loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def _assert_temporal_order(self, first: Dataset, second: Dataset, raise_on_error: bool = True) -> bool:
        base_dataset_first, first_indices = self._resolve_indices(first)
        base_dataset_second, second_indices = self._resolve_indices(second)

        if base_dataset_first is not base_dataset_second:
            raise ValueError("Temporal assertions require datasets derived from the same base dataset")

        start_ts = base_dataset_first.get_start_timestamps()
        horizon_ts = base_dataset_first.get_horizon_timestamps()

        max_first = horizon_ts[first_indices].max() if first_indices.size else pd.Timestamp.min.tz_localize("UTC")
        min_second = start_ts[second_indices].min() if second_indices.size else pd.Timestamp.max.tz_localize("UTC")
        valid = max_first < min_second
        if not valid and raise_on_error:
            raise ValueError("Temporal ordering violated: potential leakage detected")
        return valid

    # ----------------------------
    # Validation strategies
    # ----------------------------
    def static_holdout(
        self,
        cutoff: pd.Timestamp,
        *,
        val_size: float = 0.2,
        shuffle_train: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        starts = self.dataset.get_start_timestamps()
        horizons = self.dataset.get_horizon_timestamps()

        train_val_mask = horizons < cutoff
        test_mask = starts >= cutoff

        train_val_indices = np.where(train_val_mask)[0]
        test_indices = np.where(test_mask)[0]

        train_val_ds = Subset(self.dataset, train_val_indices.tolist())
        test_ds = Subset(self.dataset, test_indices.tolist())

        if len(train_val_ds) < 2:
            raise ValueError("Not enough samples before cutoff to build train/val splits")

        buckets = self.dataset.get_vol_buckets()[train_val_indices]
        indices = np.arange(len(train_val_ds))
        if len(set(buckets)) > 1:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
            train_idx, val_idx = next(splitter.split(indices.reshape(-1, 1), buckets))
        else:
            rng = np.random.default_rng(42)
            rng.shuffle(indices)
            split = max(1, int(len(indices) * (1 - val_size)))
            train_idx, val_idx = indices[:split], indices[split:]

        train_ds = Subset(train_val_ds, train_idx.tolist())
        val_ds = Subset(train_val_ds, val_idx.tolist())

        self._assert_temporal_order(train_ds, val_ds)
        self._assert_temporal_order(train_ds, test_ds)
        self._assert_temporal_order(val_ds, test_ds)

        return (
            self._build_loader(train_ds, shuffle_train),
            self._build_loader(val_ds, False),
            self._build_loader(test_ds, False),
        )

    def walk_forward(
        self,
        train_period: pd.Timedelta,
        val_period: pd.Timedelta,
        step_size: pd.Timedelta,
        *,
        shuffle_train: bool = True,
    ) -> Generator[Tuple[DataLoader, DataLoader], None, None]:
        starts = self.dataset.get_start_timestamps()
        horizons = self.dataset.get_horizon_timestamps()

        cursor = starts.min()
        max_time = horizons.max()

        while True:
            train_start = cursor
            train_end = train_start + train_period
            val_end = train_end + val_period
            if val_end > max_time:
                break

            train_mask = (starts >= train_start) & (horizons < train_end)
            val_mask = (starts >= train_end) & (horizons < val_end)

            if not train_mask.any() or not val_mask.any():
                cursor += step_size
                continue

            train_ds = Subset(self.dataset, np.where(train_mask)[0].tolist())
            val_ds = Subset(self.dataset, np.where(val_mask)[0].tolist())
            if not self._assert_temporal_order(train_ds, val_ds, raise_on_error=False):
                cursor += step_size
                continue

            yield (
                self._build_loader(train_ds, shuffle_train),
                self._build_loader(val_ds, False),
            )

            cursor += step_size

    def hybrid_nested(
        self,
        holdout_fraction: float,
        train_period: pd.Timedelta,
        val_period: pd.Timedelta,
        step_size: pd.Timedelta,
        *,
        shuffle_train: bool = True,
    ) -> Tuple[Generator[Tuple[DataLoader, DataLoader], None, None], DataLoader]:
        if not 0 < holdout_fraction < 1:
            raise ValueError("holdout_fraction must be between 0 and 1")

        decisions = self.dataset.get_decision_timestamps()
        horizons = self.dataset.get_horizon_timestamps()
        min_time = decisions.min()
        max_time = decisions.max()
        holdout_start = max_time - (max_time - min_time) * holdout_fraction

        holdout_mask = decisions >= holdout_start
        playground_mask = decisions < holdout_start

        holdout_ds = Subset(self.dataset, np.where(holdout_mask)[0].tolist())
        holdout_loader = self._build_loader(holdout_ds, False)

        starts = self.dataset.get_start_timestamps()

        def generator() -> Generator[Tuple[DataLoader, DataLoader], None, None]:
            cursor = starts.min()
            while True:
                train_start = cursor
                train_end = train_start + train_period
                val_end = train_end + val_period
                if val_end >= holdout_start:
                    break

                train_mask = playground_mask & (starts >= train_start) & (horizons < train_end)
                val_mask = playground_mask & (starts >= train_end) & (horizons < val_end)

                if not train_mask.any() or not val_mask.any():
                    cursor += step_size
                    continue

                train_ds = Subset(self.dataset, np.where(train_mask)[0].tolist())
                val_ds = Subset(self.dataset, np.where(val_mask)[0].tolist())
                if not self._assert_temporal_order(train_ds, val_ds, raise_on_error=False):
                    cursor += step_size
                    continue

                yield (
                    self._build_loader(train_ds, shuffle_train),
                    self._build_loader(val_ds, False),
                )

                cursor += step_size

        return generator(), holdout_loader


if __name__ == "__main__":
    source = MockDataSource(length=2000, freq="15min", seed=123)
    engineer = WaveletEngineer(wavelet="db4", level=3)
    loader = MarketDataLoader(
        data_source=source,
        engineer=engineer,
        assets=["BTC", "ETH"],
        input_len=64,
        pred_len=16,
        batch_size=16,
    )

    train_period = pd.Timedelta(days=10)
    val_period = pd.Timedelta(days=2)
    step_size = pd.Timedelta(days=2)

    hybrid_generator, holdout_loader = loader.hybrid_nested(
        holdout_fraction=0.1,
        train_period=train_period,
        val_period=val_period,
        step_size=step_size,
    )

    print("Hybrid playground fragments:")
    for idx, (train_dl, val_dl) in enumerate(hybrid_generator):
        print(f"Fragment {idx}: train batches={len(train_dl)}, val batches={len(val_dl)}")
        if idx >= 2:
            break
    print(f"Holdout batches={len(holdout_loader)}")
