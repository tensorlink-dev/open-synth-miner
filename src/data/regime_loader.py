"""Regime-aware flexible data loader with balanced sampling.

Seven components:

1. ``aggregate_5m_to_1h``  -- proper OHLCV aggregation with intra-hour stats
2. ``engineer_features``   -- log-ret + realized vol + Parkinson vol + intra-hour
3. ``RegimeTagger``        -- KMeans on 2-D vol manifold (train-only fit)
4. ``generate_walk_forward_folds`` -- sliding window with gap buffers
5. ``RegimeBalancedSampler``       -- temporal-order-preserving regime weighting
6. ``RegimeDriftMonitor``  -- symmetric KL trigger for retraining
7. ``run_pipeline``        -- end-to-end orchestrator
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
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

from src.data.market_data_loader import (
    AssetData,
    DataSource,
    FeatureEngineer,
    MockDataSource,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. aggregate_5m_to_1h
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. engineer_features
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 3. RegimeTagger
# ---------------------------------------------------------------------------


class RegimeTagger:
    """KMeans-based regime labeller on the 2-D vol manifold.

    Clusters are defined by ``(realized_vol, parkinson_vol)`` — the two
    dimensions that capture the volatility manifold at intrinsic
    dimensionality ~3.

    The tagger **must** be fitted exclusively on training data for each
    fold to avoid look-ahead bias.

    Parameters
    ----------
    n_regimes:
        Number of KMeans clusters (default 3).
    random_state:
        Seed for reproducibility.
    vol_features:
        Column names to use for clustering. Defaults to
        ``["realized_vol", "parkinson_vol"]``.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        random_state: int = 42,
        vol_features: Optional[List[str]] = None,
    ) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.vol_features = vol_features or ["realized_vol", "parkinson_vol"]
        self._kmeans: Optional[KMeans] = None
        self._scaler: Optional[StandardScaler] = None

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None

    def fit(self, features: pd.DataFrame) -> "RegimeTagger":
        """Fit KMeans on the training partition only.

        Parameters
        ----------
        features:
            DataFrame with at least the columns listed in ``vol_features``.
        """
        X = features[self.vol_features].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._kmeans = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=10,
        )
        self._kmeans.fit(X_scaled)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Assign regime labels to each row.

        Returns
        -------
        np.ndarray
            Integer regime label per row, shape ``(len(features),)``.
        """
        if self._kmeans is None or self._scaler is None:
            raise RuntimeError("RegimeTagger must be fit() before predict()")
        X = features[self.vol_features].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self._scaler.transform(X)
        return self._kmeans.predict(X_scaled)

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
        counts = np.bincount(labels, minlength=self.n_regimes).astype(np.float64)
        total = counts.sum()
        if total == 0:
            return np.ones(self.n_regimes) / self.n_regimes
        return counts / total


# ---------------------------------------------------------------------------
# 4. generate_walk_forward_folds
# ---------------------------------------------------------------------------


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

    # Underlying hourly bar DataFrames (with timestamps as index)
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
    n_regimes: int = 3,
    regime_random_state: int = 42,
    vol_features: Optional[List[str]] = None,
) -> List[Fold]:
    """Generate chronological walk-forward folds with gap buffers.

    For each fold the layout is::

        |--- train ---|-- gap --|--- val ---|-- gap --|--- test ---|

    The :class:`RegimeTagger` is fitted **only** on the training partition
    of each fold, then used to label all three partitions. This prevents
    any look-ahead contamination.

    Parameters
    ----------
    features:
        Feature DataFrame from :func:`engineer_features`.  Must have a
        ``DatetimeIndex`` or an integer ``RangeIndex``.
    train_size, val_size, test_size, step_size, gap_size:
        Either integer row-counts or ``pd.Timedelta`` objects.  When a
        Timedelta is provided the feature index must be a DatetimeIndex.
    n_regimes:
        Clusters for the per-fold RegimeTagger.
    regime_random_state:
        Seed forwarded to each RegimeTagger.
    vol_features:
        Columns for regime clustering (defaults handled by RegimeTagger).

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
        # Estimate row count from median frequency
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

        # Fit regime tagger on training data only
        tagger = RegimeTagger(
            n_regimes=n_regimes,
            random_state=regime_random_state,
            vol_features=vol_features,
        )
        train_labels = tagger.fit_predict(train_df)
        val_labels = tagger.predict(val_df)
        test_labels = tagger.predict(test_df)

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


# ---------------------------------------------------------------------------
# 5. RegimeBalancedSampler
# ---------------------------------------------------------------------------


class RegimeBalancedSampler(Sampler[int]):
    """Sampler that preserves temporal order while balancing regime exposure.

    For each epoch the sampler selects *start indices* of contiguous
    subsequences of length ``seq_len``.  The probability of picking a
    particular start index is a blend of uniform and inverse-frequency
    weighting based on the regime label at that position:

    .. math::

        p_i = (1 - s) \\cdot \\frac{1}{N} + s \\cdot \\frac{w_{r_i}}{\\sum_j w_{r_j}}

    where :math:`s` is ``balance_strength`` and :math:`w_r = 1 / freq(r)`.

    Within each drawn subsequence, temporal order is **preserved** (no
    shuffling inside the window), which is essential for sequence models
    like TimeMixer.

    Parameters
    ----------
    regime_labels:
        Integer array of regime labels, one per time step.
    seq_len:
        Length of contiguous subsequences to draw.
    n_samples:
        How many subsequences per epoch.  Defaults to the number of
        valid start positions.
    balance_strength:
        Float in ``[0, 1]``.  0 = uniform sampling, 1 = fully balanced
        (minority regimes maximally boosted).
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

        # Valid start positions: must have seq_len + pred_len bars after start
        n_total = len(self.regime_labels)
        total_window = seq_len + pred_len
        max_start = n_total - total_window
        if max_start < 0:
            raise ValueError(
                f"seq_len + pred_len ({total_window}) exceeds data length ({n_total})"
            )
        self._valid_starts = np.arange(0, max_start + 1)
        self._n_samples = n_samples or len(self._valid_starts)

        # Build sampling weights
        self._weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """Compute per-start-index sampling probabilities."""
        n_regimes = int(self.regime_labels.max()) + 1
        counts = np.bincount(self.regime_labels, minlength=n_regimes).astype(np.float64)
        total = counts.sum()
        freq = counts / max(total, 1.0)
        # Inverse frequency weights per regime
        inv_freq = np.where(freq > 0, 1.0 / freq, 0.0)

        # Weight for each valid start = inv_freq of the regime at that position
        start_regimes = self.regime_labels[self._valid_starts]
        regime_weights = inv_freq[start_regimes]

        # Normalize regime component
        regime_sum = regime_weights.sum()
        if regime_sum > 0:
            regime_probs = regime_weights / regime_sum
        else:
            regime_probs = np.ones(len(self._valid_starts)) / len(self._valid_starts)

        # Uniform component
        uniform_probs = np.ones(len(self._valid_starts)) / len(self._valid_starts)

        # Blend
        s = self.balance_strength
        blended = (1 - s) * uniform_probs + s * regime_probs

        # Renormalize (numerical safety)
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
        # Sort to encourage temporal locality in batches (DataLoader can
        # still shuffle batches if desired).
        yield from chosen.tolist()

    def __len__(self) -> int:
        return self._n_samples

    def effective_distribution(self) -> Dict[int, float]:
        """Report the effective sampling probability per regime.

        Useful for verifying the minority-boost behavior, e.g. a 22.6%
        minority regime being boosted to ~42% sampling probability.
        """
        n_regimes = int(self.regime_labels.max()) + 1
        regime_probs = np.zeros(n_regimes, dtype=np.float64)
        start_regimes = self.regime_labels[self._valid_starts]
        for r in range(n_regimes):
            mask = start_regimes == r
            regime_probs[r] = self._weights[mask].sum()
        return {int(r): float(p) for r, p in enumerate(regime_probs)}

    def summary(self) -> str:
        """Human-readable summary of regime balancing."""
        n_regimes = int(self.regime_labels.max()) + 1
        counts = np.bincount(self.regime_labels, minlength=n_regimes)
        total = counts.sum()
        eff = self.effective_distribution()

        lines = [f"RegimeBalancedSampler (strength={self.balance_strength:.2f})"]
        for r in range(n_regimes):
            data_pct = 100.0 * counts[r] / max(total, 1)
            samp_pct = 100.0 * eff[r]
            boost = samp_pct / max(data_pct, 1e-8)
            lines.append(
                f"  Regime {r}: {data_pct:5.1f}% data -> "
                f"{samp_pct:5.1f}% sampling ({boost:.2f}x)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. RegimeDriftMonitor
# ---------------------------------------------------------------------------


class RegimeDriftMonitor:
    """Detects regime distribution drift via symmetric KL divergence.

    Monitors a rolling window of live vol features against the training-time
    regime distribution and fires a retraining signal when the distributions
    diverge beyond a threshold.

    Parameters
    ----------
    tagger:
        A **fitted** :class:`RegimeTagger` (from the most recent training fold).
    train_distribution:
        The empirical regime distribution from training data
        (shape ``(n_regimes,)``).
    kl_threshold:
        Symmetric KL divergence threshold above which ``should_retrain``
        returns ``True``.  Default 0.5.
    window_size:
        Number of recent observations to consider (default 168 = 1 week
        of hourly bars).
    """

    def __init__(
        self,
        tagger: RegimeTagger,
        train_distribution: np.ndarray,
        kl_threshold: float = 0.5,
        window_size: int = 168,
    ) -> None:
        if not tagger.is_fitted:
            raise ValueError("RegimeDriftMonitor requires a fitted RegimeTagger")
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

        # Trim to window_size
        while self._total_rows > self.window_size and len(self._buffer) > 1:
            removed = self._buffer.pop(0)
            self._total_rows -= len(removed)

    @property
    def current_kl(self) -> float:
        """Most recently computed symmetric KL divergence."""
        return self._last_kl

    def should_retrain(self) -> bool:
        """Return ``True`` if the regime distribution has drifted.

        Computes symmetric KL divergence between the training distribution
        and the live window distribution::

            D_sym(P || Q) = 0.5 * KL(P || Q) + 0.5 * KL(Q || P)
        """
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


# ---------------------------------------------------------------------------
# Dataset adapter for the regime-balanced pipeline
# ---------------------------------------------------------------------------


class RegimeAwareDataset(Dataset):
    """Wraps a feature DataFrame + regime labels into a PyTorch Dataset.

    Each item is a contiguous window of ``seq_len`` feature rows starting
    at the index provided by the sampler.  Targets are the log returns
    for the ``pred_len`` bars immediately following the input window.

    Parameters
    ----------
    features:
        Feature matrix from :func:`engineer_features`, shape ``(T, F)``.
    regime_labels:
        Integer regime labels, shape ``(T,)``.
    seq_len:
        Input context length.
    pred_len:
        Prediction horizon.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        regime_labels: np.ndarray,
        seq_len: int,
        pred_len: int,
    ) -> None:
        self.feature_values = features.values.astype(np.float32)
        self.feature_columns = list(features.columns)
        self.regime_labels = np.asarray(regime_labels, dtype=int)
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Pre-compute log returns for targets
        close_idx = self.feature_columns.index("close")
        close = self.feature_values[:, close_idx].astype(np.float64)
        log_close = np.log(np.clip(close, 1e-12, None))
        self.log_returns = np.diff(log_close, prepend=log_close[0]).astype(np.float32)

        max_start = len(self.feature_values) - seq_len - pred_len
        if max_start < 0:
            raise ValueError("Data too short for the requested seq_len + pred_len")
        self._max_start = max_start

    def __len__(self) -> int:
        return self._max_start + 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # idx is a start position (provided by RegimeBalancedSampler or
        # sequential access).
        start = idx
        inp_end = start + self.seq_len
        tgt_end = inp_end + self.pred_len

        inputs = torch.from_numpy(
            self.feature_values[start:inp_end]
        ).float()  # (seq_len, F)

        targets = torch.from_numpy(
            self.log_returns[inp_end:tgt_end]
        ).float()  # (pred_len,)

        # Close price at decision boundary (for path simulation)
        close_idx = self.feature_columns.index("close")
        initial_price = float(self.feature_values[inp_end - 1, close_idx])

        regime = int(self.regime_labels[start])

        return {
            "inputs": inputs.T,  # (F, seq_len) to match existing convention
            "target": targets.unsqueeze(0),  # (1, pred_len)
            "initial_price": torch.tensor(initial_price, dtype=torch.float32),
            "regime": regime,
            "start_idx": start,
        }


# ---------------------------------------------------------------------------
# 7. run_pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the regime-aware data pipeline."""

    # Aggregation
    resample_rule: str = "1h"

    # Feature engineering
    realized_vol_window: int = 24
    parkinson_vol_window: int = 24
    smooth_window: int = 6

    # Regime tagging
    n_regimes: int = 3
    regime_random_state: int = 42
    vol_features: Optional[List[str]] = None

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
    # Fit tagger on training data (already done inside generate_walk_forward_folds
    # but we re-fit here to get the tagger object for monitoring)
    tagger = RegimeTagger(
        n_regimes=config.n_regimes,
        random_state=config.regime_random_state,
        vol_features=config.vol_features,
    )
    train_labels = tagger.fit_predict(fold.train_features)
    val_labels = tagger.predict(fold.val_features)
    test_labels = tagger.predict(fold.test_features)

    # Build datasets
    train_ds = RegimeAwareDataset(
        fold.train_features, train_labels, config.seq_len, config.pred_len
    )
    val_ds = RegimeAwareDataset(
        fold.val_features, val_labels, config.seq_len, config.pred_len
    )
    test_ds = RegimeAwareDataset(
        fold.test_features, test_labels, config.seq_len, config.pred_len
    )

    # Build balanced sampler for training
    sampler = RegimeBalancedSampler(
        regime_labels=train_labels,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        n_samples=config.n_samples_per_epoch,
        balance_strength=config.balance_strength,
        seed=config.regime_random_state + fold.fold_idx,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Build drift monitor keyed to training distribution
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
    """End-to-end regime-aware data pipeline.

    1. Aggregate raw sub-hourly OHLCV candles into hourly bars
    2. Engineer features (log_ret, realized vol, Parkinson vol, intra-hour)
    3. Generate walk-forward folds with gap buffers
    4. For each fold, build regime-balanced DataLoaders + drift monitor

    Parameters
    ----------
    df_raw:
        Raw OHLCV DataFrame with ``DatetimeIndex`` and columns
        ``[open, high, low, close, volume]``.
    config:
        Pipeline configuration (uses defaults if ``None``).

    Returns
    -------
    list[FoldLoaders]
        One entry per walk-forward fold, each containing train/val/test
        DataLoaders, the balanced sampler, drift monitor, and tagger.

    Example
    -------
    ::

        import pandas as pd
        from src.data.regime_loader import run_pipeline, PipelineConfig

        # Load 5-min candles
        df = pd.read_parquet("btc_5m.parquet")
        df.index = pd.to_datetime(df["timestamp"], utc=True)

        cfg = PipelineConfig(
            balance_strength=0.8,
            seq_len=64,
            pred_len=12,
        )

        fold_loaders = run_pipeline(df, cfg)
        for fl in fold_loaders:
            print(fl.sampler.summary())
            for batch in fl.train_loader:
                # batch["inputs"]: (B, F, seq_len)
                # batch["target"]:  (B, 1, pred_len)
                ...
    """
    if config is None:
        config = PipelineConfig()

    # Step 1: aggregate
    logger.info("Aggregating raw candles with rule=%s", config.resample_rule)
    df_1h = aggregate_5m_to_1h(df_raw, resample_rule=config.resample_rule)
    logger.info("Aggregated to %d hourly bars", len(df_1h))

    # Step 2: engineer features
    features = engineer_features(
        df_1h,
        realized_vol_window=config.realized_vol_window,
        parkinson_vol_window=config.parkinson_vol_window,
        smooth_window=config.smooth_window,
    )
    logger.info("Engineered %d features x %d rows", features.shape[1], features.shape[0])

    # Step 3: walk-forward folds
    folds = generate_walk_forward_folds(
        features,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        step_size=config.step_size,
        gap_size=config.gap_size,
        n_regimes=config.n_regimes,
        regime_random_state=config.regime_random_state,
        vol_features=config.vol_features,
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


# ---------------------------------------------------------------------------
# Convenience: build from AssetData (integrates with existing sources)
# ---------------------------------------------------------------------------


def asset_to_ohlcv_frame(asset: AssetData) -> pd.DataFrame:
    """Convert an :class:`AssetData` record into a DatetimeIndex OHLCV frame.

    This bridges the existing ``DataSource`` / ``AssetData`` abstraction
    to the raw DataFrame expected by :func:`aggregate_5m_to_1h`.
    """
    if asset.covariates is None or asset.covariate_columns is None:
        # Fallback: treat prices as close, set O=H=L=close, vol=1
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
    """Run the regime-aware pipeline for each asset from an existing DataSource.

    Returns
    -------
    dict[str, list[FoldLoaders]]
        Mapping from asset name to its list of fold loaders.
    """
    assets_data = source.load_data(assets)
    results: Dict[str, List[FoldLoaders]] = {}

    for asset in assets_data:
        logger.info("Processing asset: %s (%d raw bars)", asset.name, len(asset.prices))
        df_raw = asset_to_ohlcv_frame(asset)
        fold_loaders = run_pipeline(df_raw, config)
        results[asset.name] = fold_loaders

    return results


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

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

    cfg = PipelineConfig(
        train_size=200,
        val_size=50,
        test_size=50,
        step_size=50,
        gap_size=10,
        seq_len=32,
        pred_len=8,
        balance_strength=0.8,
        batch_size=16,
    )

    fold_loaders = run_pipeline(df_raw, cfg)

    for fl in fold_loaders:
        print(f"\n--- Fold {fl.fold_idx} ---")
        print(fl.sampler.summary())
        print(f"Monitor: {fl.monitor.status()}")

        for batch in fl.train_loader:
            print(f"  Batch inputs: {batch['inputs'].shape}, targets: {batch['target'].shape}")
            break
