"""Regime-aware flexible data loader with balanced sampling.

Extension points (ABCs):

- :class:`ClusteringStrategy`  -- pluggable clustering (KMeans, GMM, HDBSCAN, …)
- :class:`FeatureStep`         -- pluggable feature engineering pipeline
- :class:`TargetBuilder`       -- pluggable target extraction for any model type
- :class:`AggregationStep`     -- pluggable bar aggregation

Concrete components:

1. ``aggregate_5m_to_1h``  -- proper OHLCV aggregation with intra-hour stats
2. ``engineer_features``   -- log-ret + realized vol + Parkinson vol + intra-hour
3. ``RegimeTagger``        -- clustering on arbitrary features (train-only fit)
4. ``generate_walk_forward_folds`` -- sliding window with gap buffers
5. ``RegimeBalancedSampler``       -- temporal-order-preserving label weighting
6. ``RegimeDriftMonitor``  -- symmetric KL trigger for retraining
7. ``run_pipeline``        -- composable end-to-end orchestrator
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Sampler

from osa.data.market_data_loader import (
    AssetData,
    DataSource,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Extension point: ClusteringStrategy
# ===================================================================


class ClusteringStrategy(abc.ABC):
    """Abstract clustering backend for regime detection.

    Subclass this to plug in any unsupervised clustering algorithm
    (GMM, HDBSCAN, spectral clustering, etc.).

    The strategy operates on a pre-scaled 2-D numpy array and returns
    integer cluster labels.
    """

    @abc.abstractmethod
    def fit(self, X: np.ndarray) -> "ClusteringStrategy":
        """Fit the clusterer on training data ``X`` of shape ``(N, D)``."""

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer labels for ``X`` of shape ``(N, D)``."""

    @property
    @abc.abstractmethod
    def n_clusters(self) -> int:
        """Number of distinct clusters the strategy produces."""


class KMeansStrategy(ClusteringStrategy):
    """KMeans clustering (the default).

    Parameters
    ----------
    n_clusters:
        Number of clusters.
    random_state:
        Seed for reproducibility.
    n_init:
        Number of KMeans initialisations.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        n_init: int = 10,
    ) -> None:
        self._n_clusters = n_clusters
        self._random_state = random_state
        self._n_init = n_init
        self._model: Optional[KMeans] = None

    def fit(self, X: np.ndarray) -> "KMeansStrategy":
        self._model = KMeans(
            n_clusters=self._n_clusters,
            random_state=self._random_state,
            n_init=self._n_init,
        )
        self._model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("KMeansStrategy must be fit() before predict()")
        return self._model.predict(X)

    @property
    def n_clusters(self) -> int:
        return self._n_clusters


class GaussianMixtureStrategy(ClusteringStrategy):
    """Gaussian Mixture Model clustering.

    Parameters
    ----------
    n_components:
        Number of mixture components.
    random_state:
        Seed for reproducibility.
    covariance_type:
        Type of covariance parameters (``"full"``, ``"tied"``, ``"diag"``,
        ``"spherical"``).
    """

    def __init__(
        self,
        n_components: int = 3,
        random_state: int = 42,
        covariance_type: str = "full",
    ) -> None:
        self._n_components = n_components
        self._random_state = random_state
        self._covariance_type = covariance_type
        self._model: Any = None

    def fit(self, X: np.ndarray) -> "GaussianMixtureStrategy":
        from sklearn.mixture import GaussianMixture

        self._model = GaussianMixture(
            n_components=self._n_components,
            random_state=self._random_state,
            covariance_type=self._covariance_type,
        )
        self._model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("GaussianMixtureStrategy must be fit() before predict()")
        return self._model.predict(X)

    @property
    def n_clusters(self) -> int:
        return self._n_components


# ===================================================================
# Extension point: FeatureStep
# ===================================================================


class FeatureStep(abc.ABC):
    """Abstract feature engineering pipeline.

    A ``FeatureStep`` transforms aggregated bar data into a feature
    DataFrame suitable for regime tagging and model consumption.

    Subclass this to define custom feature sets — e.g. momentum factors,
    order-flow imbalance, cross-asset correlations, etc.
    """

    @abc.abstractmethod
    def transform(self, df_bars: pd.DataFrame) -> pd.DataFrame:
        """Transform aggregated bars into a feature DataFrame.

        Parameters
        ----------
        df_bars:
            DataFrame with at minimum ``[open, high, low, close, volume]``
            columns and a ``DatetimeIndex``.

        Returns
        -------
        pd.DataFrame
            Feature matrix.  Must contain all columns needed downstream
            (clustering features, model inputs, target source column).
        """

    @property
    @abc.abstractmethod
    def feature_names(self) -> List[str]:
        """Ordered list of output column names."""


class OHLCVFeatureStep(FeatureStep):
    """Default feature step: volatility + intra-hour stats.

    Produces 13 features: OHLCV + log_ret + realized_vol + parkinson_vol
    + skew + kurtosis + 3 smoothed intra-hour stats.

    Parameters
    ----------
    realized_vol_window:
        Lookback for realized volatility (default 24).
    parkinson_vol_window:
        Lookback for Parkinson vol (default 24).
    smooth_window:
        EMA span for smoothing intra-hour features (default 6).
    """

    def __init__(
        self,
        realized_vol_window: int = 24,
        parkinson_vol_window: int = 24,
        smooth_window: int = 6,
    ) -> None:
        self.realized_vol_window = realized_vol_window
        self.parkinson_vol_window = parkinson_vol_window
        self.smooth_window = smooth_window

    @property
    def feature_names(self) -> List[str]:
        return list(HOURLY_FEATURE_NAMES)

    def transform(self, df_bars: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(
            df_bars,
            realized_vol_window=self.realized_vol_window,
            parkinson_vol_window=self.parkinson_vol_window,
            smooth_window=self.smooth_window,
        )


# ===================================================================
# Extension point: AggregationStep
# ===================================================================


class AggregationStep(abc.ABC):
    """Abstract bar aggregation step.

    Transforms raw sub-bar candles into aggregated bars.
    """

    @abc.abstractmethod
    def aggregate(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Aggregate raw bars into target-frequency bars.

        Parameters
        ----------
        df_raw:
            Raw OHLCV DataFrame with a ``DatetimeIndex``.

        Returns
        -------
        pd.DataFrame
            Aggregated bars.
        """


class OHLCVAggregation(AggregationStep):
    """Default OHLCV aggregation with intra-hour stats.

    Parameters
    ----------
    resample_rule:
        Pandas frequency string (default ``"1h"``).
    """

    def __init__(self, resample_rule: str = "1h") -> None:
        self.resample_rule = resample_rule

    def aggregate(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        return aggregate_5m_to_1h(df_raw, resample_rule=self.resample_rule)


# ===================================================================
# Extension point: TargetBuilder
# ===================================================================


class TargetBuilder(abc.ABC):
    """Abstract target extraction strategy.

    Defines how to build training targets from a feature DataFrame.
    Subclass this for different prediction tasks: raw returns,
    multi-step forecasts, classification targets, etc.
    """

    @abc.abstractmethod
    def build_targets(self, features: pd.DataFrame) -> np.ndarray:
        """Pre-compute the full target array.

        Parameters
        ----------
        features:
            The feature DataFrame (same as model inputs).

        Returns
        -------
        np.ndarray
            1-D float32 array of length ``len(features)``.
        """

    @abc.abstractmethod
    def extract_target(
        self, targets: np.ndarray, start: int, length: int
    ) -> torch.Tensor:
        """Slice a target window and return a Tensor.

        Parameters
        ----------
        targets:
            Full pre-computed target array.
        start:
            Start index of the target window.
        length:
            Number of steps.

        Returns
        -------
        torch.Tensor
            Target tensor ready for loss computation.
        """

    def extract_initial_price(
        self,
        features: np.ndarray,
        feature_columns: List[str],
        decision_idx: int,
    ) -> float:
        """Extract the decision-boundary price for path simulation.

        Override if your model doesn't use close price.
        """
        if "close" in feature_columns:
            close_idx = feature_columns.index("close")
            return float(features[decision_idx, close_idx])
        return 0.0


class LogReturnTarget(TargetBuilder):
    """Default: log returns of the close price.

    Parameters
    ----------
    price_column:
        Column name to derive log returns from (default ``"close"``).
    """

    def __init__(self, price_column: str = "close") -> None:
        self.price_column = price_column

    def build_targets(self, features: pd.DataFrame) -> np.ndarray:
        if self.price_column not in features.columns:
            raise ValueError(
                f"Target column '{self.price_column}' not in features. "
                f"Available: {list(features.columns)}"
            )
        close = features[self.price_column].values.astype(np.float64)
        log_close = np.log(np.clip(close, 1e-12, None))
        returns = np.diff(log_close, prepend=log_close[0]).astype(np.float32)
        returns[~np.isfinite(returns)] = 0.0
        return returns

    def extract_target(
        self, targets: np.ndarray, start: int, length: int
    ) -> torch.Tensor:
        window = targets[start : start + length]
        return torch.from_numpy(window).float().unsqueeze(0)  # (1, length)


class RawReturnTarget(TargetBuilder):
    """Raw (non-log) percentage returns.

    Parameters
    ----------
    price_column:
        Column name to derive returns from (default ``"close"``).
    """

    def __init__(self, price_column: str = "close") -> None:
        self.price_column = price_column

    def build_targets(self, features: pd.DataFrame) -> np.ndarray:
        close = features[self.price_column].values.astype(np.float64)
        returns = np.diff(close, prepend=close[0]) / np.clip(
            np.concatenate([[close[0]], close[:-1]]), 1e-12, None
        )
        return returns.astype(np.float32)

    def extract_target(
        self, targets: np.ndarray, start: int, length: int
    ) -> torch.Tensor:
        window = targets[start : start + length]
        return torch.from_numpy(window).float().unsqueeze(0)  # (1, length)


class MultiColumnTarget(TargetBuilder):
    """Multi-column target for models that predict several quantities.

    Parameters
    ----------
    columns:
        List of column names to include as targets.
    """

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self._feature_columns: Optional[List[str]] = None

    def build_targets(self, features: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.columns if c not in features.columns]
        if missing:
            raise ValueError(f"Target columns {missing} not in features")
        self._feature_columns = list(features.columns)
        return features[self.columns].values.astype(np.float32)

    def extract_target(
        self, targets: np.ndarray, start: int, length: int
    ) -> torch.Tensor:
        window = targets[start : start + length]  # (length, n_cols)
        return torch.from_numpy(window).float().T  # (n_cols, length)


# ===================================================================
# 1. aggregate_5m_to_1h
# ===================================================================

_AGG_INTRA_COLS = [
    "intra_ret_std",
    "intra_max_drawdown",
    "intra_directional_consistency",
]

HOURLY_FEATURE_NAMES: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "realized_vol",
    "skew",
    "kurtosis",
    "parkinson_vol",
    "log_ret",
    # intra-hour stats
    "intra_ret_std",
    "intra_max_drawdown",
    "intra_directional_consistency",
]


def aggregate_5m_to_1h(
    df_raw: pd.DataFrame,
    resample_rule: str = "1h",
) -> pd.DataFrame:
    """Aggregate sub-hourly OHLCV candles into hourly bars.

    Proper aggregation rules:
    - open  -> first
    - high  -> max
    - low   -> min
    - close -> last
    - volume -> sum

    Plus three intra-hour statistics computed on the raw bars *within*
    each hourly window:
    - ``intra_ret_std``: std of 5-min log returns inside the hour
    - ``intra_max_drawdown``: worst peak-to-trough drawdown within the hour
    - ``intra_directional_consistency``: fraction of sub-bars whose return
      sign matches the hourly bar direction

    Parameters
    ----------
    df_raw:
        DataFrame with a ``DatetimeIndex`` and columns
        ``[open, high, low, close, volume]``.
    resample_rule:
        Pandas frequency string for the target bar size (default ``"1h"``).

    Returns
    -------
    pd.DataFrame
        Hourly bars with OHLCV + derived features.
    """
    df = df_raw.copy()

    # Pre-compute on raw bars
    df["log_close"] = np.log(df["close"].clip(lower=1e-12))
    df["log_ret"] = df["log_close"].diff()

    # Intra-bar drawdown helper: cumulative return inside each hour
    df["cum_ret"] = df["log_ret"].fillna(0.0)

    resampler = df.resample(resample_rule)

    # --- Core OHLCV aggregation (NOT .last()!) ---
    agg = resampler.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # --- Intra-hour statistics ---
    def _intra_ret_std(series: pd.Series) -> float:
        vals = series.dropna()
        return float(vals.std()) if len(vals) > 1 else 0.0

    def _intra_max_drawdown(series: pd.Series) -> float:
        vals = series.dropna().values
        if len(vals) < 2:
            return 0.0
        cumulative = np.cumsum(vals)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    def _directional_consistency(group: pd.DataFrame) -> float:
        rets = group["log_ret"].dropna()
        if len(rets) < 1:
            return 0.5
        hourly_dir = rets.sum()
        if abs(hourly_dir) < 1e-12:
            return 0.5
        sign_match = (np.sign(rets) == np.sign(hourly_dir)).sum()
        return float(sign_match / len(rets))

    intra_std = resampler["log_ret"].apply(_intra_ret_std)
    intra_dd = resampler["log_ret"].apply(_intra_max_drawdown)
    intra_dc = resampler.apply(_directional_consistency)

    agg["intra_ret_std"] = intra_std
    agg["intra_max_drawdown"] = intra_dd
    agg["intra_directional_consistency"] = intra_dc

    agg = agg.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return agg


# ===================================================================
# 2. engineer_features
# ===================================================================


def engineer_features(
    df_1h: pd.DataFrame,
    realized_vol_window: int = 24,
    parkinson_vol_window: int = 24,
    smooth_window: int = 6,
) -> pd.DataFrame:
    """Build the feature matrix from hourly bars.

    Core features (applied to true hourly bars):
    - ``log_ret``: log return of close
    - ``realized_vol``: rolling std of log returns
    - ``parkinson_vol``: rolling Parkinson volatility estimator

    Supplementary features (smoothed intra-hour stats for regime
    discrimination):
    - EMA-smoothed ``intra_ret_std``
    - EMA-smoothed ``intra_max_drawdown``
    - EMA-smoothed ``intra_directional_consistency``

    Also computes ``skew`` and ``kurtosis`` of log returns.

    Parameters
    ----------
    df_1h:
        Output of :func:`aggregate_5m_to_1h`.
    realized_vol_window:
        Lookback for realized volatility (default 24 = 1 day of hourly bars).
    parkinson_vol_window:
        Lookback for Parkinson vol (default 24).
    smooth_window:
        EMA span for smoothing intra-hour features (default 6).

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by hourly timestamps, columns matching
        ``HOURLY_FEATURE_NAMES``.
    """
    out = df_1h[["open", "high", "low", "close", "volume"]].copy()

    log_close = np.log(out["close"].clip(lower=1e-12))
    log_ret = log_close.diff().fillna(0.0)
    out["log_ret"] = log_ret

    # Realized volatility: rolling std
    out["realized_vol"] = log_ret.rolling(
        realized_vol_window, min_periods=1
    ).std().fillna(0.0)

    # Parkinson volatility: sqrt(mean(ln(H/L)^2) / (4*ln2))
    park_sq = np.log(out["high"].clip(lower=1e-12) / out["low"].clip(lower=1e-12)) ** 2
    const_4ln2 = 4.0 * np.log(2.0)
    out["parkinson_vol"] = np.sqrt(
        park_sq.rolling(parkinson_vol_window, min_periods=1).mean().fillna(0.0) / const_4ln2
    )

    # Higher moments
    out["skew"] = log_ret.rolling(realized_vol_window, min_periods=3).apply(
        lambda x: sp_skew(x, nan_policy="omit"), raw=True
    ).fillna(0.0)
    out["kurtosis"] = log_ret.rolling(realized_vol_window, min_periods=3).apply(
        lambda x: sp_kurtosis(x, nan_policy="omit"), raw=True
    ).fillna(0.0)

    # Smoothed intra-hour features
    for col in _AGG_INTRA_COLS:
        if col in df_1h.columns:
            out[col] = df_1h[col].ewm(span=smooth_window, min_periods=1).mean()
        else:
            out[col] = 0.0

    out = out[HOURLY_FEATURE_NAMES]
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


# ===================================================================
# 3. RegimeTagger
# ===================================================================


class RegimeTagger:
    """Pluggable regime labeller with customisable clustering backend.

    Fits a clustering model on selected columns of the feature DataFrame
    using a :class:`ClusteringStrategy`.  The tagger **must** be fitted
    exclusively on training data for each fold to avoid look-ahead bias.

    Parameters
    ----------
    n_regimes:
        Number of clusters (forwarded to the strategy if applicable).
    random_state:
        Seed for reproducibility.
    vol_features:
        Column names to use for clustering. Defaults to
        ``["realized_vol", "parkinson_vol"]``.
    clustering:
        A :class:`ClusteringStrategy` instance.  Defaults to
        :class:`KMeansStrategy`.  Pass ``GaussianMixtureStrategy()``,
        or your own subclass for custom behaviour.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        random_state: int = 42,
        vol_features: Optional[List[str]] = None,
        clustering: Optional[ClusteringStrategy] = None,
    ) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.vol_features = vol_features or ["realized_vol", "parkinson_vol"]
        self._clustering = clustering or KMeansStrategy(
            n_clusters=n_regimes, random_state=random_state
        )
        self._scaler: Optional[StandardScaler] = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _extract_and_scale(
        self, features: pd.DataFrame, fit_scaler: bool = False
    ) -> np.ndarray:
        """Extract clustering columns and scale."""
        X = features[self.vol_features].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if fit_scaler:
            self._scaler = StandardScaler()
            return self._scaler.fit_transform(X)
        if self._scaler is None:
            raise RuntimeError("RegimeTagger must be fit() before predict()")
        return self._scaler.transform(X)

    def fit(self, features: pd.DataFrame) -> "RegimeTagger":
        """Fit the clustering backend on the training partition only."""
        X_scaled = self._extract_and_scale(features, fit_scaler=True)
        self._clustering.fit(X_scaled)
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Assign regime labels to each row.

        Returns
        -------
        np.ndarray
            Integer regime label per row, shape ``(len(features),)``.
        """
        if not self._fitted:
            raise RuntimeError("RegimeTagger must be fit() before predict()")
        X_scaled = self._extract_and_scale(features)
        return self._clustering.predict(X_scaled)

    def fit_predict(self, features: pd.DataFrame) -> np.ndarray:
        """Convenience: fit on features then return labels."""
        self.fit(features)
        return self.predict(features)

    def regime_distribution(self, labels: np.ndarray) -> np.ndarray:
        """Return empirical distribution over regimes.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_regimes,)`` summing to 1.
        """
        n = self._clustering.n_clusters
        counts = np.bincount(labels, minlength=n).astype(np.float64)
        total = counts.sum()
        if total == 0:
            return np.ones(n) / n
        return counts / total


# ===================================================================
# 4. generate_walk_forward_folds
# ===================================================================


@dataclass
class Fold:
    """A single walk-forward fold with data and regime labels."""

    fold_idx: int

    train_features: pd.DataFrame
    val_features: pd.DataFrame
    test_features: pd.DataFrame

    train_regimes: np.ndarray
    val_regimes: np.ndarray
    test_regimes: np.ndarray

    # Underlying bar DataFrames (with timestamps as index)
    train_bars: pd.DataFrame
    val_bars: pd.DataFrame
    test_bars: pd.DataFrame


def generate_walk_forward_folds(
    features: pd.DataFrame,
    *,
    train_size: Union[int, pd.Timedelta],
    val_size: Union[int, pd.Timedelta],
    test_size: Union[int, pd.Timedelta],
    step_size: Union[int, pd.Timedelta],
    gap_size: Union[int, pd.Timedelta] = 0,
    tagger: Optional[RegimeTagger] = None,
    # Legacy convenience params (used when tagger is None)
    n_regimes: int = 3,
    regime_random_state: int = 42,
    vol_features: Optional[List[str]] = None,
) -> List[Fold]:
    """Generate chronological walk-forward folds with gap buffers.

    For each fold the layout is::

        |--- train ---|-- gap --|--- val ---|-- gap --|--- test ---|

    The tagger is fitted **only** on the training partition of each fold,
    then used to label all three partitions.  This prevents any
    look-ahead contamination.

    Parameters
    ----------
    features:
        Feature DataFrame.  Must have a ``DatetimeIndex`` or integer index.
    train_size, val_size, test_size, step_size, gap_size:
        Either integer row-counts or ``pd.Timedelta`` objects.
    tagger:
        A :class:`RegimeTagger` instance to clone per fold.  If ``None``,
        one is created from ``n_regimes`` / ``regime_random_state`` /
        ``vol_features``.
    n_regimes, regime_random_state, vol_features:
        Legacy convenience params used when ``tagger`` is ``None``.

    Returns
    -------
    list[Fold]
        Ordered list of non-overlapping folds.
    """
    is_datetime = isinstance(features.index, pd.DatetimeIndex)

    def _to_rows(size: Union[int, pd.Timedelta]) -> int:
        if isinstance(size, int):
            return size
        if not is_datetime:
            raise TypeError(
                "Timedelta sizes require a DatetimeIndex on the features DataFrame"
            )
        freq = features.index.to_series().diff().dropna().median()
        return max(1, int(size / freq))

    train_n = _to_rows(train_size)
    val_n = _to_rows(val_size)
    test_n = _to_rows(test_size)
    step_n = _to_rows(step_size)
    gap_n = _to_rows(gap_size) if not isinstance(gap_size, int) or gap_size != 0 else int(gap_size)

    total = len(features)
    window = train_n + gap_n + val_n + gap_n + test_n

    if window > total:
        raise ValueError(
            f"Total fold window ({window} rows) exceeds data length ({total} rows)"
        )

    folds: List[Fold] = []
    cursor = 0
    fold_idx = 0

    while cursor + window <= total:
        t_start = cursor
        t_end = t_start + train_n
        v_start = t_end + gap_n
        v_end = v_start + val_n
        s_start = v_end + gap_n
        s_end = s_start + test_n

        train_df = features.iloc[t_start:t_end]
        val_df = features.iloc[v_start:v_end]
        test_df = features.iloc[s_start:s_end]

        # Build a fresh tagger per fold (clone settings, never reuse fitted state)
        if tagger is not None:
            fold_tagger = RegimeTagger(
                n_regimes=tagger.n_regimes,
                random_state=tagger.random_state,
                vol_features=tagger.vol_features,
                clustering=type(tagger._clustering)(
                    **_clone_strategy_params(tagger._clustering)
                ) if hasattr(tagger._clustering, '__init__') else None,
            )
        else:
            fold_tagger = RegimeTagger(
                n_regimes=n_regimes,
                random_state=regime_random_state,
                vol_features=vol_features,
            )

        train_labels = fold_tagger.fit_predict(train_df)
        val_labels = fold_tagger.predict(val_df)
        test_labels = fold_tagger.predict(test_df)

        folds.append(
            Fold(
                fold_idx=fold_idx,
                train_features=train_df,
                val_features=val_df,
                test_features=test_df,
                train_regimes=train_labels,
                val_regimes=val_labels,
                test_regimes=test_labels,
                train_bars=train_df,
                val_bars=val_df,
                test_bars=test_df,
            )
        )

        cursor += step_n
        fold_idx += 1

    if not folds:
        raise ValueError("No folds could be generated with the given parameters")

    return folds


def _clone_strategy_params(strategy: ClusteringStrategy) -> Dict[str, Any]:
    """Extract init params from a clustering strategy for cloning."""
    if isinstance(strategy, KMeansStrategy):
        return {
            "n_clusters": strategy._n_clusters,
            "random_state": strategy._random_state,
            "n_init": strategy._n_init,
        }
    if isinstance(strategy, GaussianMixtureStrategy):
        return {
            "n_components": strategy._n_components,
            "random_state": strategy._random_state,
            "covariance_type": strategy._covariance_type,
        }
    # Fallback: try to extract n_clusters
    return {"n_clusters": strategy.n_clusters}


# ===================================================================
# 5. RegimeBalancedSampler
# ===================================================================


class RegimeBalancedSampler(Sampler[int]):
    """Sampler that preserves temporal order while balancing label exposure.

    Works with *any* integer label array — not just regime labels.
    Could balance by volatility bucket, sector, time-of-day, etc.

    For each epoch the sampler selects *start indices* of contiguous
    subsequences.  The probability of picking a particular start index is
    a blend of uniform and inverse-frequency weighting based on the label
    at that position:

    .. math::

        p_i = (1 - s) \\cdot \\frac{1}{N} + s \\cdot \\frac{w_{r_i}}{\\sum_j w_{r_j}}

    where :math:`s` is ``balance_strength`` and :math:`w_r = 1 / freq(r)`.

    Within each drawn subsequence, temporal order is **preserved**.

    Parameters
    ----------
    regime_labels:
        Integer array of labels, one per time step.
    seq_len:
        Length of contiguous subsequences to draw.
    pred_len:
        Additional look-ahead needed after seq_len (for targets).
    n_samples:
        How many subsequences per epoch.
    balance_strength:
        Float in ``[0, 1]``.  0 = uniform, 1 = fully balanced.
    seed:
        Random seed.
    """

    def __init__(
        self,
        regime_labels: np.ndarray,
        seq_len: int,
        pred_len: int = 0,
        n_samples: Optional[int] = None,
        balance_strength: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.regime_labels = np.asarray(regime_labels, dtype=int)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.balance_strength = np.clip(balance_strength, 0.0, 1.0)
        self.seed = seed

        n_total = len(self.regime_labels)
        total_window = seq_len + pred_len
        max_start = n_total - total_window
        if max_start < 0:
            raise ValueError(
                f"seq_len + pred_len ({total_window}) exceeds data length ({n_total})"
            )
        self._valid_starts = np.arange(0, max_start + 1)
        self._n_samples = n_samples or len(self._valid_starts)

        self._weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """Compute per-start-index sampling probabilities."""
        n_labels = int(self.regime_labels.max()) + 1
        counts = np.bincount(self.regime_labels, minlength=n_labels).astype(np.float64)
        total = counts.sum()
        freq = counts / max(total, 1.0)
        inv_freq = np.where(freq > 0, 1.0 / freq, 0.0)

        start_labels = self.regime_labels[self._valid_starts]
        label_weights = inv_freq[start_labels]

        label_sum = label_weights.sum()
        if label_sum > 0:
            label_probs = label_weights / label_sum
        else:
            label_probs = np.ones(len(self._valid_starts)) / len(self._valid_starts)

        uniform_probs = np.ones(len(self._valid_starts)) / len(self._valid_starts)

        s = self.balance_strength
        blended = (1 - s) * uniform_probs + s * label_probs
        blended /= blended.sum()
        return blended

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        chosen = rng.choice(
            self._valid_starts,
            size=self._n_samples,
            replace=True,
            p=self._weights,
        )
        yield from chosen.tolist()

    def __len__(self) -> int:
        return self._n_samples

    def effective_distribution(self) -> Dict[int, float]:
        """Report effective sampling probability per label."""
        n_labels = int(self.regime_labels.max()) + 1
        label_probs = np.zeros(n_labels, dtype=np.float64)
        start_labels = self.regime_labels[self._valid_starts]
        for r in range(n_labels):
            mask = start_labels == r
            label_probs[r] = self._weights[mask].sum()
        return {int(r): float(p) for r, p in enumerate(label_probs)}

    def summary(self) -> str:
        """Human-readable summary of label balancing."""
        n_labels = int(self.regime_labels.max()) + 1
        counts = np.bincount(self.regime_labels, minlength=n_labels)
        total = counts.sum()
        eff = self.effective_distribution()

        lines = [f"RegimeBalancedSampler (strength={self.balance_strength:.2f})"]
        for r in range(n_labels):
            data_pct = 100.0 * counts[r] / max(total, 1)
            samp_pct = 100.0 * eff[r]
            boost = samp_pct / max(data_pct, 1e-8)
            lines.append(
                f"  Regime {r}: {data_pct:5.1f}% data -> "
                f"{samp_pct:5.1f}% sampling ({boost:.2f}x)"
            )
        return "\n".join(lines)


# ===================================================================
# 6. RegimeDriftMonitor
# ===================================================================


class RegimeDriftMonitor:
    """Detects label distribution drift via symmetric KL divergence.

    Works with any object that has ``predict(df) -> labels`` and
    ``regime_distribution(labels) -> dist`` methods — not coupled to
    :class:`RegimeTagger` specifically.

    Parameters
    ----------
    tagger:
        A **fitted** tagger with ``predict`` and ``regime_distribution``.
    train_distribution:
        Empirical distribution from training (shape ``(n_labels,)``).
    kl_threshold:
        Symmetric KL divergence threshold for ``should_retrain``.
    window_size:
        Rolling window of recent observations.
    """

    def __init__(
        self,
        tagger: Any,
        train_distribution: np.ndarray,
        kl_threshold: float = 0.5,
        window_size: int = 168,
    ) -> None:
        if hasattr(tagger, "is_fitted") and not tagger.is_fitted:
            raise ValueError("RegimeDriftMonitor requires a fitted tagger")
        self.tagger = tagger
        self.train_dist = np.asarray(train_distribution, dtype=np.float64)
        self.kl_threshold = kl_threshold
        self.window_size = window_size
        self._buffer: List[pd.DataFrame] = []
        self._total_rows = 0
        self._last_kl: float = 0.0

    def update(self, new_features: pd.DataFrame) -> None:
        """Append new observations to the rolling buffer."""
        self._buffer.append(new_features)
        self._total_rows += len(new_features)

        while self._total_rows > self.window_size and len(self._buffer) > 1:
            removed = self._buffer.pop(0)
            self._total_rows -= len(removed)

    @property
    def current_kl(self) -> float:
        """Most recently computed symmetric KL divergence."""
        return self._last_kl

    def should_retrain(self) -> bool:
        """Return ``True`` if the label distribution has drifted."""
        if self._total_rows < max(10, self.window_size // 4):
            return False

        window_df = pd.concat(self._buffer, axis=0).tail(self.window_size)
        live_labels = self.tagger.predict(window_df)
        live_dist = self.tagger.regime_distribution(live_labels)

        self._last_kl = self._symmetric_kl(self.train_dist, live_dist)
        return self._last_kl > self.kl_threshold

    @staticmethod
    def _symmetric_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """Compute symmetric KL divergence with smoothing."""
        p = np.clip(p, eps, None)
        q = np.clip(q, eps, None)
        p = p / p.sum()
        q = q / q.sum()
        kl_pq = float(np.sum(p * np.log(p / q)))
        kl_qp = float(np.sum(q * np.log(q / p)))
        return 0.5 * kl_pq + 0.5 * kl_qp

    def status(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot."""
        window_df = pd.concat(self._buffer, axis=0).tail(self.window_size) if self._buffer else pd.DataFrame()
        live_labels = self.tagger.predict(window_df) if len(window_df) > 0 else np.array([])
        live_dist = self.tagger.regime_distribution(live_labels) if len(live_labels) > 0 else np.zeros_like(self.train_dist)

        return {
            "train_distribution": self.train_dist.tolist(),
            "live_distribution": live_dist.tolist(),
            "symmetric_kl": self._last_kl,
            "threshold": self.kl_threshold,
            "should_retrain": self._last_kl > self.kl_threshold,
            "buffer_rows": self._total_rows,
        }


# ===================================================================
# Dataset adapter
# ===================================================================


class RegimeAwareDataset(Dataset):
    """Flexible dataset wrapping features + labels + pluggable targets.

    Each item is a contiguous window of ``seq_len`` feature rows.
    Target extraction is delegated to a :class:`TargetBuilder`.

    Parameters
    ----------
    features:
        Feature DataFrame, shape ``(T, F)``.
    regime_labels:
        Integer labels, shape ``(T,)``.
    seq_len:
        Input context length.
    pred_len:
        Prediction horizon.
    target_builder:
        Strategy for target extraction.  Defaults to
        :class:`LogReturnTarget`.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        regime_labels: np.ndarray,
        seq_len: int,
        pred_len: int,
        target_builder: Optional[TargetBuilder] = None,
    ) -> None:
        self.feature_values = features.values.astype(np.float32)
        self.feature_columns = list(features.columns)
        self.regime_labels = np.asarray(regime_labels, dtype=int)
        self.seq_len = seq_len
        self.pred_len = pred_len

        self._target_builder = target_builder or LogReturnTarget()
        self._targets = self._target_builder.build_targets(features)

        max_start = len(self.feature_values) - seq_len - pred_len
        if max_start < 0:
            raise ValueError("Data too short for the requested seq_len + pred_len")
        self._max_start = max_start

    def __len__(self) -> int:
        return self._max_start + 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start = idx
        inp_end = start + self.seq_len
        tgt_end = inp_end + self.pred_len

        inputs = torch.from_numpy(
            self.feature_values[start:inp_end]
        ).float()  # (seq_len, F)

        targets = self._target_builder.extract_target(
            self._targets, inp_end, self.pred_len
        )

        initial_price = self._target_builder.extract_initial_price(
            self.feature_values, self.feature_columns, inp_end - 1
        )

        regime = int(self.regime_labels[start])

        return {
            "inputs": inputs.T,  # (F, seq_len) to match existing convention
            "target": targets,
            "initial_price": torch.tensor(initial_price, dtype=torch.float32),
            "regime": regime,
            "start_idx": start,
        }


# ===================================================================
# 7. run_pipeline
# ===================================================================


@dataclass
class PipelineConfig:
    """Configuration for the regime-aware data pipeline.

    All extension points have sensible defaults that reproduce the
    original OHLCV + KMeans + log-return behaviour.  Override any
    combination to customise:

    - ``aggregation``: custom bar aggregation
    - ``feature_step``: custom feature engineering
    - ``clustering``: custom clustering backend (GMM, HDBSCAN, …)
    - ``target_builder``: custom target extraction
    - ``vol_features``: which columns to cluster on
    """

    # Aggregation
    resample_rule: str = "1h"
    aggregation: Optional[AggregationStep] = None

    # Feature engineering
    realized_vol_window: int = 24
    parkinson_vol_window: int = 24
    smooth_window: int = 6
    feature_step: Optional[FeatureStep] = None

    # Regime tagging
    n_regimes: int = 3
    regime_random_state: int = 42
    vol_features: Optional[List[str]] = None
    clustering: Optional[ClusteringStrategy] = None

    # Walk-forward folds
    train_size: Union[int, pd.Timedelta] = 720  # 30 days of hourly bars
    val_size: Union[int, pd.Timedelta] = 168    # 7 days
    test_size: Union[int, pd.Timedelta] = 168   # 7 days
    step_size: Union[int, pd.Timedelta] = 168   # slide by 7 days
    gap_size: Union[int, pd.Timedelta] = 24     # 1 day buffer

    # Sampler
    seq_len: int = 64
    pred_len: int = 12
    balance_strength: float = 0.8
    batch_size: int = 32
    n_samples_per_epoch: Optional[int] = None

    # Target
    target_builder: Optional[TargetBuilder] = None

    # Drift monitor
    kl_threshold: float = 0.5
    drift_window_size: int = 168


@dataclass
class FoldLoaders:
    """DataLoader triplet for a single fold, plus diagnostics."""

    fold_idx: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    sampler: RegimeBalancedSampler
    monitor: RegimeDriftMonitor
    tagger: RegimeTagger
    fold: Fold


def _build_fold_loaders(
    fold: Fold,
    config: PipelineConfig,
) -> FoldLoaders:
    """Build DataLoaders for one walk-forward fold."""
    tagger = RegimeTagger(
        n_regimes=config.n_regimes,
        random_state=config.regime_random_state,
        vol_features=config.vol_features,
        clustering=config.clustering,
    )
    train_labels = tagger.fit_predict(fold.train_features)
    val_labels = tagger.predict(fold.val_features)
    test_labels = tagger.predict(fold.test_features)

    tb = config.target_builder

    train_ds = RegimeAwareDataset(
        fold.train_features, train_labels, config.seq_len, config.pred_len,
        target_builder=tb,
    )
    val_ds = RegimeAwareDataset(
        fold.val_features, val_labels, config.seq_len, config.pred_len,
        target_builder=tb,
    )
    test_ds = RegimeAwareDataset(
        fold.test_features, test_labels, config.seq_len, config.pred_len,
        target_builder=tb,
    )

    sampler = RegimeBalancedSampler(
        regime_labels=train_labels,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        n_samples=config.n_samples_per_epoch,
        balance_strength=config.balance_strength,
        seed=config.regime_random_state + fold.fold_idx,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=sampler, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, drop_last=False,
    )

    train_dist = tagger.regime_distribution(train_labels)
    monitor = RegimeDriftMonitor(
        tagger=tagger,
        train_distribution=train_dist,
        kl_threshold=config.kl_threshold,
        window_size=config.drift_window_size,
    )

    return FoldLoaders(
        fold_idx=fold.fold_idx,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        sampler=sampler,
        monitor=monitor,
        tagger=tagger,
        fold=fold,
    )


def run_pipeline(
    df_raw: pd.DataFrame,
    config: Optional[PipelineConfig] = None,
) -> List[FoldLoaders]:
    """End-to-end composable data pipeline.

    Each step is pluggable via ``PipelineConfig``:

    1. **Aggregation** — ``config.aggregation`` or default OHLCV
    2. **Feature engineering** — ``config.feature_step`` or default OHLCV features
    3. **Walk-forward folds** with regime tagging
    4. **Balanced DataLoaders** with custom targets

    Parameters
    ----------
    df_raw:
        Raw OHLCV DataFrame with ``DatetimeIndex``.
    config:
        Pipeline configuration.

    Returns
    -------
    list[FoldLoaders]

    Examples
    --------
    Default (OHLCV + KMeans + log returns)::

        fold_loaders = run_pipeline(df_raw)

    Custom clustering + features::

        from sklearn.mixture import GaussianMixture

        cfg = PipelineConfig(
            clustering=GaussianMixtureStrategy(n_components=4),
            vol_features=["realized_vol", "skew", "kurtosis"],
            target_builder=RawReturnTarget(),
        )
        fold_loaders = run_pipeline(df_raw, cfg)

    Custom feature step::

        class MyFeatures(FeatureStep):
            @property
            def feature_names(self):
                return ["close", "momentum", "rsi"]
            def transform(self, df_bars):
                # your feature logic here
                ...

        cfg = PipelineConfig(
            feature_step=MyFeatures(),
            vol_features=["momentum"],  # cluster on momentum
        )
    """
    if config is None:
        config = PipelineConfig()

    # Step 1: aggregate
    agg_step = config.aggregation or OHLCVAggregation(
        resample_rule=config.resample_rule
    )
    logger.info("Aggregating raw candles")
    df_agg = agg_step.aggregate(df_raw)
    logger.info("Aggregated to %d bars", len(df_agg))

    # Step 2: engineer features
    feat_step = config.feature_step or OHLCVFeatureStep(
        realized_vol_window=config.realized_vol_window,
        parkinson_vol_window=config.parkinson_vol_window,
        smooth_window=config.smooth_window,
    )
    features = feat_step.transform(df_agg)
    logger.info("Engineered %d features x %d rows", features.shape[1], features.shape[0])

    # Step 3: walk-forward folds
    tagger_template = RegimeTagger(
        n_regimes=config.n_regimes,
        random_state=config.regime_random_state,
        vol_features=config.vol_features,
        clustering=config.clustering,
    )
    folds = generate_walk_forward_folds(
        features,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        step_size=config.step_size,
        gap_size=config.gap_size,
        tagger=tagger_template,
    )
    logger.info("Generated %d walk-forward folds", len(folds))

    # Step 4: build loaders per fold
    fold_loaders: List[FoldLoaders] = []
    for fold in folds:
        fl = _build_fold_loaders(fold, config)
        logger.info(
            "Fold %d: %s",
            fold.fold_idx,
            fl.sampler.summary().replace("\n", " | "),
        )
        fold_loaders.append(fl)

    return fold_loaders


# ===================================================================
# Convenience: build from AssetData
# ===================================================================


def asset_to_ohlcv_frame(asset: AssetData) -> pd.DataFrame:
    """Convert an :class:`AssetData` record into a DatetimeIndex OHLCV frame."""
    if asset.covariates is None or asset.covariate_columns is None:
        n = len(asset.prices)
        close = np.asarray(asset.prices, dtype=np.float64)
        close = np.clip(close, 1e-12, None)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.ones(n, dtype=np.float64),
            },
            index=pd.DatetimeIndex(pd.to_datetime(asset.timestamps, utc=True)),
        )
        return df

    cov_map = {name: idx for idx, name in enumerate(asset.covariate_columns)}
    needed = {"open", "high", "low", "volume"}
    if not needed.issubset(cov_map):
        raise ValueError(
            f"AssetData for {asset.name} missing required covariates "
            f"{needed - set(cov_map)}"
        )

    n = len(asset.prices)
    close = np.clip(np.asarray(asset.prices[:n], dtype=np.float64), 1e-12, None)
    ts = pd.DatetimeIndex(pd.to_datetime(asset.timestamps[:n], utc=True))

    df = pd.DataFrame(
        {
            "open": np.asarray(asset.covariates[:n, cov_map["open"]], dtype=np.float64),
            "high": np.asarray(asset.covariates[:n, cov_map["high"]], dtype=np.float64),
            "low": np.asarray(asset.covariates[:n, cov_map["low"]], dtype=np.float64),
            "close": close,
            "volume": np.asarray(asset.covariates[:n, cov_map["volume"]], dtype=np.float64),
        },
        index=ts,
    )
    return df


def run_pipeline_from_source(
    source: DataSource,
    assets: List[str],
    config: Optional[PipelineConfig] = None,
) -> Dict[str, List[FoldLoaders]]:
    """Run the pipeline for each asset from an existing DataSource."""
    assets_data = source.load_data(assets)
    results: Dict[str, List[FoldLoaders]] = {}

    for asset in assets_data:
        logger.info("Processing asset: %s (%d raw bars)", asset.name, len(asset.prices))
        df_raw = asset_to_ohlcv_frame(asset)
        fold_loaders = run_pipeline(df_raw, config)
        results[asset.name] = fold_loaders

    return results


# ===================================================================
# Demo / smoke test
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Generate synthetic 5-min candles
    np.random.seed(42)
    n_bars = 5000
    timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    log_rets = rng.normal(0, 0.001, n_bars)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n_bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n_bars)))
    open_ = close * (1 + rng.normal(0, 0.0005, n_bars))
    volume = np.abs(rng.normal(1000, 300, n_bars))

    df_raw = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=timestamps,
    )

    # --- Demo 1: Default pipeline (backward compatible) ---
    print("=== Default Pipeline (KMeans + LogReturn) ===")
    cfg = PipelineConfig(
        train_size=200, val_size=50, test_size=50,
        step_size=50, gap_size=10,
        seq_len=32, pred_len=8,
        balance_strength=0.8, batch_size=16,
    )
    fold_loaders = run_pipeline(df_raw, cfg)
    for fl in fold_loaders[:2]:
        print(f"\n--- Fold {fl.fold_idx} ---")
        print(fl.sampler.summary())
        for batch in fl.train_loader:
            print(f"  inputs: {batch['inputs'].shape}, target: {batch['target'].shape}")
            break

    # --- Demo 2: GMM clustering + raw returns ---
    print("\n=== GMM Pipeline (GaussianMixture + RawReturn) ===")
    cfg2 = PipelineConfig(
        train_size=200, val_size=50, test_size=50,
        step_size=50, gap_size=10,
        seq_len=32, pred_len=8,
        clustering=GaussianMixtureStrategy(n_components=4),
        vol_features=["realized_vol", "parkinson_vol", "skew"],
        target_builder=RawReturnTarget(),
        balance_strength=0.8, batch_size=16,
    )
    fold_loaders2 = run_pipeline(df_raw, cfg2)
    for fl in fold_loaders2[:2]:
        print(f"\n--- Fold {fl.fold_idx} ---")
        print(fl.sampler.summary())
        for batch in fl.train_loader:
            print(f"  inputs: {batch['inputs'].shape}, target: {batch['target'].shape}")
            break
