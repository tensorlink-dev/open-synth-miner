"""Ablation grid generator for systematic architecture search.

Generates a cross-product of model configurations varying:
- Feature engineers (ZScore, Wavelet)
- RevIN / DLinear block presence
- DLinear kernel sizes
- Model heads (GBM, SDE, SimpleHorizon, CLTHorizon, GaussianSpectral)

Each combination is emitted as a named OmegaConf config compatible with
:class:`AblationExperiment`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Axis specifications
# ---------------------------------------------------------------------------

ENGINEER_SPECS: Dict[str, Dict[str, Any]] = {
    "zscore": {
        "_target_": "src.data.market_data_loader.ZScoreEngineer",
        "short_win": 20,
        "long_win": 200,
        "feature_dim": 3,
    },
    "wavelet": {
        "_target_": "src.data.market_data_loader.WaveletEngineer",
        "wavelet": "db4",
        "level": 4,
        "feature_dim": 5,
    },
}

# (use_revin, use_dlinear) label → booleans
REVIN_DLINEAR_COMBOS: Dict[str, Tuple[bool, bool]] = {
    "none": (False, False),
    "revin": (True, False),
    "dlinear": (False, True),
    "revin_dlinear": (True, True),
}

DEFAULT_KERNEL_SIZES: List[int] = [15, 25, 51]

# Head specs: name → (_target_, extra kwargs)
# Only heads that accept just latent_size (or simple extras) are included
# to keep the grid runnable without special routing logic.
HEAD_SPECS: Dict[str, Dict[str, Any]] = {
    "gbm": {"_target_": "src.models.heads.GBMHead"},
    "sde": {"_target_": "src.models.heads.SDEHead"},
    "simple_horizon": {"_target_": "src.models.heads.SimpleHorizonHead"},
    "clt_horizon": {"_target_": "src.models.heads.CLTHorizonHead"},
    "gaussian_spectral": {"_target_": "src.models.heads.GaussianSpectralHead"},
}


# ---------------------------------------------------------------------------
# Grid spec
# ---------------------------------------------------------------------------


@dataclass
class AblationGridSpec:
    """Specification for which axes to sweep in an ablation study.

    Set an axis to ``None`` or an empty list to hold it fixed at the
    default value.  Every non-``None`` axis is crossed with the others.
    """

    engineers: List[str] = field(default_factory=lambda: list(ENGINEER_SPECS.keys()))
    revin_dlinear: List[str] = field(default_factory=lambda: list(REVIN_DLINEAR_COMBOS.keys()))
    kernel_sizes: List[int] = field(default_factory=lambda: list(DEFAULT_KERNEL_SIZES))
    heads: List[str] = field(default_factory=lambda: list(HEAD_SPECS.keys()))
    d_model: int = 32


# ---------------------------------------------------------------------------
# Config builder helpers
# ---------------------------------------------------------------------------


def _build_blocks(
    d_model: int,
    use_revin: bool,
    use_dlinear: bool,
    kernel_size: int,
) -> List[Dict[str, Any]]:
    """Assemble the backbone block list for one configuration."""
    blocks: List[Dict[str, Any]] = []

    if use_revin:
        blocks.append({
            "_target_": "src.models.components.advanced_blocks.RevIN",
            "d_model": d_model,
        })

    if use_dlinear:
        blocks.append({
            "_target_": "src.models.components.advanced_blocks.DLinearBlock",
            "d_model": d_model,
            "kernel_size": kernel_size,
        })

    # LSTM as the core sequence model
    blocks.append({
        "_target_": "src.models.registry.LSTMBlock",
        "d_model": d_model,
        "num_layers": 1,
    })

    return blocks


def _build_single_config(
    engineer_name: str,
    revin_dlinear_name: str,
    kernel_size: int,
    head_name: str,
    d_model: int,
    training_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a complete experiment config dict for one grid point."""
    eng_spec = ENGINEER_SPECS[engineer_name]
    feature_dim = eng_spec["feature_dim"]
    use_revin, use_dlinear = REVIN_DLINEAR_COMBOS[revin_dlinear_name]

    blocks = _build_blocks(d_model, use_revin, use_dlinear, kernel_size)

    head_cfg = {**HEAD_SPECS[head_name], "latent_size": d_model}

    # Strip feature_dim from engineer config (it's metadata, not a ctor arg)
    eng_cfg = {k: v for k, v in eng_spec.items() if k != "feature_dim"}

    cfg: Dict[str, Any] = {
        "model": {
            "_target_": "src.models.factory.SynthModel",
            "backbone": {
                "_target_": "src.models.factory.HybridBackbone",
                "input_size": feature_dim,
                "d_model": d_model,
                "validate_shapes": True,
                "blocks": blocks,
            },
            "head": head_cfg,
        },
        "data": {
            "engineer": eng_cfg,
            "feature_dim": feature_dim,
        },
        "training": {
            "batch_size": 4,
            "seq_len": 32,
            "feature_dim": feature_dim,
            "horizon": 12,
            "n_paths": 100,
            "lr": 0.001,
            "epochs": 5,
        },
    }

    if training_overrides:
        cfg["training"].update(training_overrides)

    return cfg


def _experiment_name(
    engineer: str,
    rd_combo: str,
    kernel_size: int,
    head_name: str,
    use_dlinear: bool,
) -> str:
    """Generate a human-readable experiment name."""
    parts = [f"eng={engineer}", f"blocks={rd_combo}"]
    if use_dlinear:
        parts.append(f"ks={kernel_size}")
    parts.append(f"head={head_name}")
    return "__".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_ablation_grid(
    spec: Optional[AblationGridSpec] = None,
    training_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, DictConfig]:
    """Generate the full cross-product of ablation configurations.

    Invalid combinations are automatically pruned:
    - ``kernel_size`` is only varied when DLinear is present; when DLinear
      is absent the kernel_size axis collapses to a single sentinel value.

    Parameters
    ----------
    spec:
        Axis specification.  Defaults to the full grid.
    training_overrides:
        Extra keys merged into each config's ``training`` section
        (e.g. ``{"epochs": 10, "n_paths": 200}``).

    Returns
    -------
    Dict[str, DictConfig]
        Mapping from experiment name → Hydra-compatible config.
    """
    if spec is None:
        spec = AblationGridSpec()

    configs: Dict[str, DictConfig] = {}

    for engineer, rd_combo, head_name in itertools.product(
        spec.engineers,
        spec.revin_dlinear,
        spec.heads,
    ):
        use_revin, use_dlinear = REVIN_DLINEAR_COMBOS[rd_combo]

        # Only sweep kernel_sizes when DLinear is present
        ks_values = spec.kernel_sizes if use_dlinear else [25]  # sentinel

        for ks in ks_values:
            name = _experiment_name(engineer, rd_combo, ks, head_name, use_dlinear)
            raw = _build_single_config(
                engineer_name=engineer,
                revin_dlinear_name=rd_combo,
                kernel_size=ks,
                head_name=head_name,
                d_model=spec.d_model,
                training_overrides=training_overrides,
            )
            configs[name] = OmegaConf.create(raw)

    return configs


def describe_grid(configs: Dict[str, DictConfig]) -> str:
    """Return a summary table of the ablation grid."""
    lines = [
        f"Ablation grid: {len(configs)} configurations",
        "-" * 70,
        f"{'Name':<60} {'Blocks':>6}",
        "-" * 70,
    ]
    for name, cfg in configs.items():
        n_blocks = len(cfg.model.backbone.blocks)
        lines.append(f"{name:<60} {n_blocks:>6}")
    return "\n".join(lines)
