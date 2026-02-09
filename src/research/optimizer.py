"""Auto-tuning engine for feature engineering hyperparameters.

Uses Optuna to search over feature engineer parameters and evaluates
candidates by measuring the intrinsic dimensionality of the resulting
feature space.  Lower intrinsic dimension signals a more compact,
higher-density representation which empirically leads to better
downstream model performance.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Type

import hydra
import numpy as np
import optuna
import skdim
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from src.data.market_data_loader import (
    AssetData,
    DataSource,
    FeatureEngineer,
    WaveletEngineer,
    ZScoreEngineer,
)

log = logging.getLogger(__name__)

# Registry mapping engineer type keys to their class and default feature dims
_ENGINEER_REGISTRY: Dict[str, Type[FeatureEngineer]] = {
    "zscore": ZScoreEngineer,
    "wavelet": WaveletEngineer,
}


class FeatureOptimizer:
    """Optuna-driven search over feature engineering parameters.

    For each trial the optimizer:
      1. Suggests parameters from the configured search space.
      2. Instantiates the corresponding ``FeatureEngineer``.
      3. Extracts windowed feature matrices from the loaded data.
      4. Estimates intrinsic dimensionality via TwoNN.
      5. Returns the ID estimate as the objective value (lower is better).

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.  The ``optimize`` sub-key must contain at least
        ``n_trials``, ``engineer.type``, and ``engineer.search_space``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        opt = cfg.optimize

        self.n_trials: int = opt.n_trials
        self.input_len: int = opt.get("input_len", 96)
        self.n_samples: int = opt.get("n_samples", 500)
        self.engineer_type: str = opt.engineer.type
        self.search_space: DictConfig = opt.engineer.search_space

        if self.engineer_type not in _ENGINEER_REGISTRY:
            raise ValueError(
                f"Unknown engineer type '{self.engineer_type}'. "
                f"Available: {sorted(_ENGINEER_REGISTRY)}"
            )

        # Load asset data once — reused across all trials.
        assets: List[str] = list(opt.get("assets", cfg.data.assets))
        data_source: DataSource = hydra.utils.instantiate(cfg.data.data_source)
        self.assets_data: List[AssetData] = data_source.load_data(assets)
        if not self.assets_data:
            raise ValueError("Data source returned no assets")
        log.info(
            "Loaded %d asset(s) for optimization: %s",
            len(self.assets_data),
            [a.name for a in self.assets_data],
        )

    # ------------------------------------------------------------------
    # Engineer construction helpers
    # ------------------------------------------------------------------

    def _build_engineer(self, trial: optuna.Trial) -> FeatureEngineer:
        """Instantiate a ``FeatureEngineer`` with trial-suggested params."""

        space = self.search_space

        if self.engineer_type == "zscore":
            short_win = trial.suggest_int(
                "short_win", int(space.short_win[0]), int(space.short_win[1])
            )
            long_win = trial.suggest_int(
                "long_win", int(space.long_win[0]), int(space.long_win[1])
            )
            # Ensure long > short to produce meaningful z-scores.
            if long_win <= short_win:
                long_win = short_win + 1
            return ZScoreEngineer(short_win=short_win, long_win=long_win)

        if self.engineer_type == "wavelet":
            level = trial.suggest_int(
                "level", int(space.level[0]), int(space.level[1])
            )
            wavelet = trial.suggest_categorical("wavelet", list(space.wavelet))
            return WaveletEngineer(wavelet=wavelet, level=level)

        raise ValueError(f"Unsupported engineer type: {self.engineer_type}")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, engineer: FeatureEngineer) -> np.ndarray:
        """Build a (n_windows, feature_dim * input_len) matrix."""

        all_features: List[np.ndarray] = []
        for asset in self.assets_data:
            cache = engineer.prepare_cache(asset.prices)
            n_total = len(asset.prices)
            max_start = n_total - self.input_len
            if max_start <= 0:
                continue

            # Sub-sample windows evenly across the series.
            n_windows = min(self.n_samples, max_start)
            step = max(1, max_start // n_windows)
            for start in range(0, max_start, step):
                feat = engineer.make_input(cache, start, self.input_len)
                all_features.append(feat.numpy().reshape(-1))
                if len(all_features) >= self.n_samples:
                    break

        return np.array(all_features)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: intrinsic dimensionality of feature space."""

        engineer = self._build_engineer(trial)
        X = self._extract_features(engineer)

        if X.shape[0] < 10:
            log.warning("Trial %d: too few samples (%d), returning penalty.", trial.number, X.shape[0])
            return 999.0

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            d_int = float(skdim.id.TwoNN().fit_transform(X_scaled))
        except Exception:
            log.warning("Trial %d: TwoNN estimation failed, returning penalty.", trial.number)
            return 999.0

        log.info("Trial %d  ID=%.3f  params=%s", trial.number, d_int, trial.params)
        return d_int

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full optimization study.

        Returns
        -------
        dict
            ``{"best_params": {...}, "best_value": float, "study": Study}``
        """

        sampler_name = self.cfg.optimize.get("sampler", "tpe")
        if sampler_name == "tpe":
            sampler = optuna.samplers.TPESampler(seed=42)
        elif sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=42)
        else:
            sampler = optuna.samplers.TPESampler(seed=42)

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "study": study,
        }
