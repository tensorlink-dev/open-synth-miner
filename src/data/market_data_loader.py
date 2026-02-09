"""Leak-safe market data loader with modular feature engineering."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pywt
import torch
from huggingface_hub import hf_hub_download
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew
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

    def prepare_cache_from_asset(self, asset: "AssetData") -> Any:
        """Pre-compute artifacts from the full :class:`AssetData` record.

        Override this when the engineer needs OHLCV or covariate data beyond
        the 1-D price array.  The default implementation delegates to
        :meth:`prepare_cache` for backward compatibility.
        """
        return self.prepare_cache(asset.prices)

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
    covariate_columns: Optional[List[str]] = None
    covariates: Optional[np.ndarray] = None


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
        filename: Union[str, List[str]],
        *,
        revision: Optional[str] = None,
        repo_type: Optional[str] = "dataset",
        asset_column: str = "asset",
        price_column: str = "price",
        fallback_price_column: Optional[str] = "close",
        timestamp_column: str = "timestamp",
        covariate_columns: Optional[List[str]] = None,
    ) -> None:
        self.repo_id = repo_id
        self.filenames = [filename] if isinstance(filename, str) else list(filename)
        self.revision = revision
        self.repo_type = repo_type
        self.asset_column = asset_column
        self.price_column = price_column
        self.fallback_price_column = fallback_price_column
        self.timestamp_column = timestamp_column
        self.covariate_columns = covariate_columns

    def load_data(self, assets: List[str]) -> List[AssetData]:
        frames: List[pd.DataFrame] = []
        for filename in self.filenames:
            file_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                revision=self.revision,
                repo_type=self.repo_type,
            )
            table = pq.read_table(file_path)
            frames.append(table.to_pandas())

        frame = pd.concat(frames, ignore_index=True)

        if self.timestamp_column not in frame.columns:
            raise ValueError("Parquet source must include a timestamp column")

        if self.price_column in frame.columns:
            price_series = frame[self.price_column]
        elif self.fallback_price_column and self.fallback_price_column in frame.columns:
            price_series = frame[self.fallback_price_column]
        else:
            raise ValueError("Parquet source must include a price or close column")

        frame[self.timestamp_column] = pd.to_datetime(frame[self.timestamp_column], utc=True)
        frame["_price"] = price_series

        covariate_columns = self.covariate_columns or []
        missing_covariates = [col for col in covariate_columns if col not in frame.columns]
        if missing_covariates:
            raise ValueError(f"Missing covariate columns: {missing_covariates}")

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
                        prices=subset["_price"].to_numpy(),
                        covariate_columns=covariate_columns or None,
                        covariates=subset[covariate_columns].to_numpy() if covariate_columns else None,
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
                    prices=subset["_price"].to_numpy(),
                    covariate_columns=covariate_columns or None,
                    covariates=subset[covariate_columns].to_numpy() if covariate_columns else None,
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
            cache = self.engineer.prepare_cache_from_asset(asset)
            self.caches.append(cache)

            # Use cache length (may differ from asset after resampling)
            if isinstance(cache, dict) and "returns" in cache:
                total = len(cache["returns"])
            else:
                total = len(asset.timestamps)
            max_start = total - (self.input_len + self.pred_len)
            if max_start < 0:
                continue

            # Resampled engineers store timestamps in cache
            if isinstance(cache, dict) and "timestamps" in cache:
                ts_array = cache["timestamps"]
            else:
                ts_array = asset.timestamps

            for start in range(0, max_start + 1, self.stride):
                decision_pos = start + self.input_len - 1
                horizon_pos = decision_pos + self.pred_len
                start_ts = _ts(ts_array[start])
                decision_ts = _ts(ts_array[decision_pos])
                horizon_ts = _ts(ts_array[horizon_pos])

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
            "decision_timestamp": window.decision_ts.isoformat(),
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
    stride: Optional[int] = None
    sort_on_load: bool = True
    gap_handling: str = "error"
    feature_dim: int = 3
    stride: int = 1

    def __post_init__(self) -> None:
        self.assets_data = self.data_source.load_data(self.assets)
        if not self.assets_data:
            raise ValueError("Data source returned no assets")
        self._normalize_assets()
        self.dataset = MarketDataset(
            self.assets_data,
            engineer=self.engineer,
            input_len=self.input_len,
            pred_len=self.pred_len,
            stride=self.stride,
        )
        if len(self.dataset) == 0:
            raise ValueError("No windows available for the requested input/prediction lengths")

        sample_inputs = self.dataset[0]["inputs"]
        inferred_dim = int(sample_inputs.shape[0])
        if inferred_dim != int(self.feature_dim):
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, engineer produced {inferred_dim}"
            )

    # ----------------------------
    # Utility helpers
    # ----------------------------
    def _normalize_assets(self) -> None:
        """Sort assets by timestamp and optionally fill or reject gaps."""

        valid_gap_modes = {"error", "ffill", "nan"}
        if self.gap_handling not in valid_gap_modes:
            raise ValueError(f"gap_handling must be one of {sorted(valid_gap_modes)}")

        for idx, asset in enumerate(self.assets_data):
            timestamps = pd.to_datetime(asset.timestamps, utc=True)
            prices = np.asarray(asset.prices)

            order = np.argsort(timestamps) if self.sort_on_load else np.arange(len(timestamps))
            timestamps = timestamps[order]
            prices = prices[order]
            covariates = asset.covariates[order] if asset.covariates is not None else None

            ts_index = pd.DatetimeIndex(timestamps)
            diffs = ts_index.to_series().diff().dropna()
            inferred_freq = None if diffs.empty else diffs.mode().iloc[0]

            if inferred_freq is not None:
                full_range = pd.date_range(ts_index.min(), ts_index.max(), freq=inferred_freq, tz=ts_index.tz)
                missing = full_range.difference(ts_index)
            else:
                full_range = ts_index
                missing = pd.DatetimeIndex([])

            if missing.size and self.gap_handling == "error":
                raise ValueError(
                    f"Temporal gaps detected for asset {asset.name}: {missing.size} missing observations"
                )

            if missing.size and self.gap_handling in {"ffill", "nan"}:
                columns: Dict[str, Any] = {"price": prices}
                if asset.covariates is not None:
                    covariate_cols = asset.covariate_columns or [f"cov_{i}" for i in range(covariates.shape[1])]
                    for col_idx, col_name in enumerate(covariate_cols):
                        columns[col_name] = covariates[:, col_idx]
                frame = pd.DataFrame(columns, index=ts_index)
                frame = frame.reindex(full_range)
                if self.gap_handling == "ffill":
                    frame = frame.ffill()

                prices = frame["price"].to_numpy()
                if asset.covariates is not None:
                    covariates = frame[covariate_cols].to_numpy()
                ts_index = frame.index

            self.assets_data[idx] = AssetData(
                name=asset.name,
                timestamps=ts_index.to_numpy(),
                prices=prices,
                covariate_columns=asset.covariate_columns,
                covariates=covariates,
            )

    def get_price_series(
        self,
        *,
        asset: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> torch.Tensor:
        """Return a price tensor for a given asset and optional slice.

        This provides a stable surface for backtesting flows that need raw price
        histories without duplicating loader implementations.
        """

        if asset is None:
            asset_idx = 0
        else:
            matches = {a.name: idx for idx, a in enumerate(self.assets_data)}
            if asset not in matches:
                raise ValueError(f"Unknown asset '{asset}', available assets: {list(matches)}")
            asset_idx = matches[asset]

        asset_data = self.assets_data[asset_idx]
        series = asset_data.prices
        total_len = len(series)
        start_idx = 0 if start is None else start
        end_idx = total_len if end is None else end
        if not (0 <= start_idx <= end_idx <= total_len):
            raise ValueError(f"Invalid slice [{start_idx}, {end_idx}) for series length {total_len}")

        return torch.tensor(series[start_idx:end_idx], dtype=torch.float32)

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
        cutoff: Union[pd.Timestamp, float],
        *,
        val_size: float = 0.2,
        shuffle_train: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        starts = self.dataset.get_start_timestamps()
        horizons = self.dataset.get_horizon_timestamps()

        if isinstance(cutoff, float):
            if not 0.0 < cutoff < 1.0:
                raise ValueError("Cutoff fraction must be between 0 and 1")
            cutoff_ts = pd.Series(horizons).quantile(1.0 - cutoff)
            cutoff = pd.to_datetime(cutoff_ts, utc=True)
        else:
            cutoff = pd.to_datetime(cutoff, utc=True)

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


# ---------------------------------------------------------------------------
# OHLCV-aware source and feature engineering
# ---------------------------------------------------------------------------

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


class HFOHLCVSource(DataSource):
    """Load per-asset OHLCV parquet files from Hugging Face Hub.

    The ``tensorlink-dev/open-synth-training-data`` repository stores candle
    data in per-asset folders (``BTC_USD/``, ``ETH_USD/``, …).  This source
    maps requested asset names to Hub paths and returns :class:`AssetData`
    with ``close`` as the price array and ``open / high / low / volume`` as
    covariates so that :class:`OHLCVEngineer` can access the full OHLCV record.

    Parameters
    ----------
    repo_id:
        Hugging Face dataset repository (e.g.
        ``"tensorlink-dev/open-synth-training-data"``).
    asset_files:
        Mapping from asset name to the parquet path **inside** the repo
        (e.g. ``{"BTC_USD": "BTC_USD/data.parquet"}``).  If ``None`` the
        source auto-generates paths as ``"{asset}/data.parquet"``.
    filename_pattern:
        Python format-string used when ``asset_files`` is ``None``.
        ``{asset}`` is replaced by the asset name.
        Default: ``"{asset}/data.parquet"``.
    """

    def __init__(
        self,
        repo_id: str,
        asset_files: Optional[Dict[str, str]] = None,
        *,
        filename_pattern: str = "{asset}/data.parquet",
        revision: Optional[str] = None,
        repo_type: Optional[str] = "dataset",
        timestamp_column: str = "timestamp",
        open_column: str = "open",
        high_column: str = "high",
        low_column: str = "low",
        close_column: str = "close",
        volume_column: str = "volume",
    ) -> None:
        self.repo_id = repo_id
        self.asset_files = asset_files
        self.filename_pattern = filename_pattern
        self.revision = revision
        self.repo_type = repo_type
        self.ts_col = timestamp_column
        self.open_col = open_column
        self.high_col = high_column
        self.low_col = low_column
        self.close_col = close_column
        self.vol_col = volume_column

    def _resolve_filename(self, asset: str) -> str:
        if self.asset_files and asset in self.asset_files:
            return self.asset_files[asset]
        return self.filename_pattern.format(asset=asset)

    _TIMESTAMP_ALIASES = ["timestamp", "date", "datetime", "time", "ts", "Date", "Datetime", "Timestamp"]

    def load_data(self, assets: List[str]) -> List[AssetData]:
        results: List[AssetData] = []
        for asset in assets:
            filename = self._resolve_filename(asset)
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                revision=self.revision,
                repo_type=self.repo_type,
            )
            df = pq.read_table(local_path).to_pandas()

            # Auto-detect timestamp column if the configured name is missing
            if self.ts_col not in df.columns:
                detected = None
                for alias in self._TIMESTAMP_ALIASES:
                    if alias in df.columns:
                        detected = alias
                        break
                # Fall back to a datetime-typed column
                if detected is None:
                    for col in df.columns:
                        if pd.api.types.is_datetime64_any_dtype(df[col]):
                            detected = col
                            break
                if detected is None:
                    raise ValueError(
                        f"Missing timestamp column '{self.ts_col}' in {filename}. "
                        f"Available columns: {list(df.columns)}"
                    )
                df = df.rename(columns={detected: self.ts_col})
            df[self.ts_col] = pd.to_datetime(df[self.ts_col], utc=True)
            df = df.sort_values(self.ts_col)

            col_map = {
                self.open_col: "open",
                self.high_col: "high",
                self.low_col: "low",
                self.close_col: "close",
                self.vol_col: "volume",
            }
            missing = [orig for orig in col_map if orig not in df.columns]
            if missing:
                raise ValueError(f"Missing OHLCV columns {missing} in {filename}")

            cov_cols = ["open", "high", "low", "volume"]
            results.append(
                AssetData(
                    name=asset,
                    timestamps=df[self.ts_col].to_numpy(),
                    prices=df[col_map[self.close_col]].to_numpy(dtype=np.float64),
                    covariate_columns=cov_cols,
                    covariates=df[[col_map[c] for c in [self.open_col, self.high_col, self.low_col, self.vol_col]]].to_numpy(dtype=np.float64),
                )
            )
        return results


def _engineer_features_1h(df_raw: pd.DataFrame, resample_rule: str = "1h") -> pd.DataFrame:
    """Aggregate raw OHLCV candles into 1-hour bars with micro-structure features.

    Returns a DataFrame with 16 columns (see ``OHLCV_FEATURE_NAMES``) indexed
    by the resampled timestamps.
    """

    df = df_raw.copy()

    # Pre-calculations on raw bars
    df["log_close"] = np.log(df["close"].clip(lower=1e-12))
    df["log_ret"] = df["log_close"].diff()
    df["park_sq"] = np.log(df["high"].clip(lower=1e-12) / df["low"].clip(lower=1e-12)) ** 2
    df["typ_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["pv"] = df["typ_price"] * df["volume"]
    df["path_len"] = df["close"].diff().abs()
    df["signed_vol"] = np.sign(df["close"].diff()) * df["volume"]

    resampler = df.resample(resample_rule)
    agg = resampler.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "log_ret": ["std", lambda x: sp_skew(x, nan_policy="omit"), lambda x: sp_kurtosis(x, nan_policy="omit")],
            "park_sq": "mean",
            "pv": "sum",
            "path_len": "sum",
            "signed_vol": "sum",
        }
    )

    # Flatten multi-index columns
    agg.columns = ["_".join(col).strip() if col[1] else col[0] for col in agg.columns.values]
    agg.rename(
        columns={
            "open_first": "open",
            "high_max": "high",
            "low_min": "low",
            "close_last": "close",
            "volume_sum": "volume",
            "log_ret_std": "realized_vol",
            "log_ret_<lambda_0>": "skew",
            "log_ret_<lambda_1>": "kurtosis",
        },
        inplace=True,
    )

    # Post-aggregation derived features
    const_4ln2 = 4.0 * np.log(2.0)
    agg["parkinson_vol"] = np.sqrt(agg["park_sq_mean"] / const_4ln2)

    net_move = (agg["close"] - agg["open"]).abs()
    agg["efficiency"] = net_move / (agg["path_len_sum"] + 1e-8)

    vwap = agg["pv_sum"] / (agg["volume"] + 1e-8)
    agg["vwap_dev"] = (agg["close"] - vwap) / (vwap + 1e-8)

    price_range = (agg["high"] - agg["low"]) + 1e-8
    agg["up_wick"] = (agg["high"] - agg[["open", "close"]].max(axis=1)) / price_range
    agg["down_wick"] = (agg[["open", "close"]].min(axis=1) - agg["low"]) / price_range
    agg["body_size"] = (agg["close"] - agg["open"]).abs() / price_range
    agg["clv"] = ((agg["close"] - agg["low"]) - (agg["high"] - agg["close"])) / price_range

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "realized_vol", "skew", "kurtosis", "parkinson_vol",
        "efficiency", "vwap_dev", "signed_vol_sum",
        "up_wick", "down_wick", "body_size", "clv",
    ]
    result = agg[feature_cols].copy()
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result


OHLCV_FEATURE_NAMES: List[str] = [
    "open", "high", "low", "close", "volume",
    "realized_vol", "skew", "kurtosis", "parkinson_vol",
    "efficiency", "vwap_dev", "signed_vol_sum",
    "up_wick", "down_wick", "body_size", "clv",
]


class OHLCVEngineer(FeatureEngineer):
    """Resample raw OHLCV candles to 1-hour bars with 16 micro-structure features.

    Features include Parkinson volatility, return skew/kurtosis, fractal
    efficiency, VWAP deviation, wick ratios, body dominance, and close
    location value.  The target is log-returns of the 1-hour close.

    This engineer overrides :meth:`prepare_cache_from_asset` to consume the
    full :class:`AssetData` record (OHLCV via covariates).

    Parameters
    ----------
    resample_rule:
        Pandas resample frequency string (default ``"1h"``).
    """

    def __init__(self, resample_rule: str = "1h") -> None:
        self.resample_rule = resample_rule

    # -- interface ---------------------------------------------------------

    def prepare_cache(self, prices: np.ndarray) -> Any:
        """Fallback when only a 1-D price array is available.

        Constructs a minimal OHLCV frame where O=H=L=C=price and volume=1,
        so the engineer still works with :class:`MockDataSource`.
        """
        p = self.clean_prices(prices)
        df = pd.DataFrame(
            {"open": p, "high": p, "low": p, "close": p, "volume": np.ones_like(p)},
            index=pd.RangeIndex(len(p)),
        )
        return self._cache_from_df(df)

    def prepare_cache_from_asset(self, asset: AssetData) -> Any:
        """Build the 1-hour feature cache from the full OHLCV record."""
        if asset.covariates is None or asset.covariate_columns is None:
            return self.prepare_cache(asset.prices)

        cov_map = {name: idx for idx, name in enumerate(asset.covariate_columns)}
        needed = {"open", "high", "low", "volume"}
        if not needed.issubset(cov_map):
            return self.prepare_cache(asset.prices)

        n = len(asset.prices)
        close = self.clean_prices(asset.prices)
        open_ = np.asarray(asset.covariates[:n, cov_map["open"]], dtype=np.float64)
        high = np.asarray(asset.covariates[:n, cov_map["high"]], dtype=np.float64)
        low = np.asarray(asset.covariates[:n, cov_map["low"]], dtype=np.float64)
        volume = np.asarray(asset.covariates[:n, cov_map["volume"]], dtype=np.float64)

        ts = asset.timestamps[:n]
        index = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
        return self._cache_from_df(df)

    # -- helpers -----------------------------------------------------------

    def _cache_from_df(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            df_1h = _engineer_features_1h(df, resample_rule=self.resample_rule)
        else:
            # Non-datetime index (MockDataSource fallback) — skip resample
            df_1h = self._features_no_resample(df)

        features = df_1h.values.astype(np.float32)  # (T_1h, 16)
        close_1h = df_1h["close"].values.astype(np.float64)
        log_close = np.log(np.clip(close_1h, 1e-12, None))
        returns = np.diff(log_close, prepend=log_close[0]).astype(np.float32)
        returns[~np.isfinite(returns)] = 0.0
        cache: Dict[str, np.ndarray] = {"features": features, "returns": returns}
        if isinstance(df_1h.index, pd.DatetimeIndex):
            cache["timestamps"] = df_1h.index.to_numpy()
        return cache

    @staticmethod
    def _features_no_resample(df: pd.DataFrame) -> pd.DataFrame:
        """Compute features without resampling (for non-datetime-indexed data)."""
        out = df[["open", "high", "low", "close", "volume"]].copy()
        log_close = np.log(out["close"].clip(lower=1e-12))
        log_ret = log_close.diff().fillna(0.0)

        out["realized_vol"] = log_ret.rolling(60, min_periods=1).std().fillna(0.0)
        out["skew"] = log_ret.rolling(60, min_periods=1).apply(
            lambda x: sp_skew(x, nan_policy="omit"), raw=True
        ).fillna(0.0)
        out["kurtosis"] = log_ret.rolling(60, min_periods=1).apply(
            lambda x: sp_kurtosis(x, nan_policy="omit"), raw=True
        ).fillna(0.0)

        park_sq = np.log(out["high"].clip(lower=1e-12) / out["low"].clip(lower=1e-12)) ** 2
        out["parkinson_vol"] = np.sqrt(park_sq / (4.0 * np.log(2.0)))

        net_move = (out["close"] - out["open"]).abs()
        path_len = out["close"].diff().abs().rolling(60, min_periods=1).sum().fillna(1e-8)
        out["efficiency"] = net_move / (path_len + 1e-8)

        out["vwap_dev"] = 0.0
        out["signed_vol_sum"] = (np.sign(out["close"].diff()) * out["volume"]).fillna(0.0)

        price_range = (out["high"] - out["low"]) + 1e-8
        out["up_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / price_range
        out["down_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / price_range
        out["body_size"] = (out["close"] - out["open"]).abs() / price_range
        out["clv"] = ((out["close"] - out["low"]) - (out["high"] - out["close"])) / price_range

        return out[OHLCV_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def make_input(self, cache: Any, start: int, length: int) -> torch.Tensor:
        window = cache["features"][start : start + length]
        return torch.from_numpy(window).float().T  # (16, length)

    def make_target(self, cache: Any, start: int, length: int) -> torch.Tensor:
        target = cache["returns"][start : start + length]
        return torch.from_numpy(target[None, :]).float()  # (1, length)

    def get_volatility(self, cache: Any, start: int, length: int) -> float:
        window = cache["returns"][start : start + length]
        return float(np.std(window))


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
