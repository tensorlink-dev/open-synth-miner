"""Auto-tuning engine for feature engineering hyperparameters.

Uses Optuna to search over feature engineer parameters and evaluates
candidates by measuring the intrinsic dimensionality of the resulting
feature space.  Lower intrinsic dimension signals a more compact,
higher-density representation which empirically leads to better
downstream model performance.

Custom engineers
~~~~~~~~~~~~~~~~
Any ``FeatureEngineer`` subclass can be optimized by specifying it as a
Hydra ``_target_`` in the optimize config::

    optimize:
      engineer:
        _target_: my_package.MyEngineer
        search_space:
          window:
            type: int
            low: 5
            high: 100
          alpha:
            type: float
            low: 0.01
            high: 1.0
          mode:
            type: categorical
            choices: [fast, precise]

The built-in shorthands ``type: zscore`` and ``type: wavelet`` still work
for backward compatibility.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List

import hydra
import numpy as np
import optuna
import skdim
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from src.data.market_data_loader import (
    AssetData,
    DataSource,
    FeatureEngineer,
    WaveletEngineer,
    ZScoreEngineer,
)

log = logging.getLogger(__name__)

# Shorthand aliases so users can write  type: zscore  instead of a full _target_
_SHORTHAND_TARGETS: Dict[str, str] = {
    "zscore": "src.data.market_data_loader.ZScoreEngineer",
    "wavelet": "src.data.market_data_loader.WaveletEngineer",
}


def _import_target(target: str) -> type:
    """Import a class from a dotted ``module.ClassName`` path."""
    module_path, _, class_name = target.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid _target_: '{target}' (expected module.ClassName)")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _suggest_param(trial: optuna.Trial, name: str, spec: DictConfig) -> Any:
    """Suggest a single parameter from its search-space spec."""
    ptype = str(spec["type"])

    if ptype == "int":
        return trial.suggest_int(name, int(spec.low), int(spec.high))
    if ptype == "float":
        log_scale = bool(spec.get("log", False))
        return trial.suggest_float(name, float(spec.low), float(spec.high), log=log_scale)
    if ptype == "categorical":
        return trial.suggest_categorical(name, list(spec.choices))

    raise ValueError(
        f"Unknown search_space type '{ptype}' for param '{name}'. "
        "Supported: int, float, categorical"
    )


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
        ``n_trials`` and an ``engineer`` block with either a ``_target_`` or
        a legacy ``type`` shorthand, plus a ``search_space`` mapping.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        opt = cfg.optimize

        self.n_trials: int = opt.n_trials
        self.input_len: int = opt.get("input_len", 96)
        self.n_samples: int = opt.get("n_samples", 500)

        eng_cfg = opt.engineer
        self.search_space: DictConfig = eng_cfg.search_space

        # Resolve engineer class: _target_ takes precedence, then type shorthand.
        if "_target_" in eng_cfg:
            target_str = str(eng_cfg._target_)
        elif "type" in eng_cfg:
            shorthand = str(eng_cfg.type)
            target_str = _SHORTHAND_TARGETS.get(shorthand)
            if target_str is None:
                raise ValueError(
                    f"Unknown engineer shorthand '{shorthand}'. "
                    f"Available shorthands: {sorted(_SHORTHAND_TARGETS)}. "
                    "Or use _target_ to specify any FeatureEngineer class."
                )
        else:
            raise ValueError(
                "optimize.engineer must specify either '_target_' (dotted class path) "
                "or 'type' (shorthand: zscore, wavelet)"
            )

        self.engineer_cls: type = _import_target(target_str)
        if not (isinstance(self.engineer_cls, type) and issubclass(self.engineer_cls, FeatureEngineer)):
            raise TypeError(
                f"{target_str} does not resolve to a FeatureEngineer subclass"
            )
        self.engineer_target: str = target_str
        log.info("Engineer class: %s", self.engineer_target)

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
    # Engineer construction
    # ------------------------------------------------------------------

    def _build_engineer(self, trial: optuna.Trial) -> FeatureEngineer:
        """Instantiate a ``FeatureEngineer`` with trial-suggested params."""

        kwargs: Dict[str, Any] = {}
        for name, spec in self.search_space.items():
            kwargs[name] = _suggest_param(trial, name, spec)

        # Apply any static (non-searched) params from config.
        static = self.cfg.optimize.engineer.get("static_params", {})
        if static:
            kwargs.update(OmegaConf.to_container(static, resolve=True))

        return self.engineer_cls(**kwargs)

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
