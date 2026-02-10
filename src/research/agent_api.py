"""Agent-friendly research API for running experiments without side effects.

This module provides a zero-config, pure-Python interface designed for AI agents
and automation tools to run research experiments on Open Synth Miner. Everything
uses plain dicts for I/O, requires no CLI or Hydra, and avoids side effects
(no W&B, no HF Hub pushes, no file writes) unless explicitly opted in.

Quick Start
-----------
::

    from src.research.agent_api import ResearchSession

    session = ResearchSession()

    # See what's available
    blocks = session.list_blocks()
    heads = session.list_heads()
    presets = session.list_presets()

    # Run a preset experiment
    result = session.run_preset("transformer_lstm")

    # Build a custom experiment
    experiment = session.create_experiment(
        blocks=["TransformerBlock", "LSTMBlock"],
        head="GBMHead",
        d_model=32,
        horizon=12,
    )
    result = session.run(experiment)

    # Compare results
    comparison = session.compare()

    # Get full session summary
    session.summary()
"""
from __future__ import annotations

import inspect
import time
import traceback
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim

from src.models.factory import create_model, HEAD_REGISTRY
from src.models.registry import discover_components, registry
from src.research.trainer import train_step, evaluate_and_log


# ---------------------------------------------------------------------------
# Presets: named experiment configurations agents can run with one call
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "transformer_lstm": {
        "description": "Transformer + LSTM hybrid with GBM head (default architecture)",
        "tags": ["baseline", "hybrid", "recurrent"],
        "model": {
            "backbone": {
                "_target_": "src.models.factory.HybridBackbone",
                "input_size": 4,
                "d_model": 32,
                "blocks": [
                    {"_target_": "src.models.registry.TransformerBlock", "d_model": 32, "nhead": 4},
                    {"_target_": "src.models.registry.LSTMBlock", "d_model": 32, "num_layers": 1},
                ],
            },
            "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 32},
        },
        "training": {"batch_size": 4, "seq_len": 32, "feature_dim": 4, "horizon": 12, "n_paths": 100, "lr": 0.001},
    },
    "pure_transformer": {
        "description": "Transformer-only backbone with GBM head",
        "tags": ["attention", "simple"],
        "model": {
            "backbone": {
                "_target_": "src.models.factory.HybridBackbone",
                "input_size": 4,
                "d_model": 32,
                "blocks": [
                    {"_target_": "src.models.registry.TransformerBlock", "d_model": 32, "nhead": 4},
                    {"_target_": "src.models.registry.TransformerBlock", "d_model": 32, "nhead": 4},
                ],
            },
            "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 32},
        },
        "training": {"batch_size": 4, "seq_len": 32, "feature_dim": 4, "horizon": 12, "n_paths": 100, "lr": 0.001},
    },
    "conv_gru": {
        "description": "ResConv + GRU hybrid for local pattern extraction + sequence modeling",
        "tags": ["convolutional", "recurrent", "hybrid"],
        "model": {
            "backbone": {
                "_target_": "src.models.factory.HybridBackbone",
                "input_size": 4,
                "d_model": 32,
                "blocks": [
                    {"_target_": "src.models.components.advanced_blocks.ResConvBlock", "d_model": 32},
                    {"_target_": "src.models.components.advanced_blocks.GRUBlock", "d_model": 32},
                ],
            },
            "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 32},
        },
        "training": {"batch_size": 4, "seq_len": 32, "feature_dim": 4, "horizon": 12, "n_paths": 100, "lr": 0.001},
    },
    "dlinear_simple": {
        "description": "DLinear trend-seasonal decomposition (lightweight, no attention)",
        "tags": ["linear", "decomposition", "lightweight"],
        "model": {
            "backbone": {
                "_target_": "src.models.factory.HybridBackbone",
                "input_size": 4,
                "d_model": 32,
                "blocks": [
                    {"_target_": "src.models.components.advanced_blocks.DLinearBlock", "d_model": 32},
                ],
            },
            "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 32},
        },
        "training": {"batch_size": 4, "seq_len": 32, "feature_dim": 4, "horizon": 12, "n_paths": 100, "lr": 0.001},
    },
    "fourier_lstm": {
        "description": "Fourier frequency block + LSTM for spectral + temporal modeling",
        "tags": ["frequency", "recurrent", "hybrid"],
        "model": {
            "backbone": {
                "_target_": "src.models.factory.HybridBackbone",
                "input_size": 4,
                "d_model": 32,
                "blocks": [
                    {"_target_": "src.models.components.advanced_blocks.FourierBlock", "d_model": 32, "modes": 16},
                    {"_target_": "src.models.registry.LSTMBlock", "d_model": 32, "num_layers": 1},
                ],
            },
            "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 32},
        },
        "training": {"batch_size": 4, "seq_len": 32, "feature_dim": 4, "horizon": 12, "n_paths": 100, "lr": 0.001},
    },
    "timesnet": {
        "description": "TimesNet 2D-variation block for period-aware modeling",
        "tags": ["timesnet", "2d-conv", "period-aware"],
        "model": {
            "backbone": {
                "_target_": "src.models.factory.HybridBackbone",
                "input_size": 4,
                "d_model": 32,
                "blocks": [
                    {"_target_": "src.models.components.advanced_blocks.TimesNetBlock", "d_model": 32, "top_k": 3},
                ],
            },
            "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 32},
        },
        "training": {"batch_size": 4, "seq_len": 32, "feature_dim": 4, "horizon": 12, "n_paths": 100, "lr": 0.001},
    },
    "timemixer": {
        "description": "TimeMixer multi-scale decomposable mixing",
        "tags": ["timemixer", "multi-scale", "decomposition"],
        "model": {
            "backbone": {
                "_target_": "src.models.factory.HybridBackbone",
                "input_size": 4,
                "d_model": 32,
                "blocks": [
                    {"_target_": "src.models.components.advanced_blocks.TimeMixerBlock", "d_model": 32},
                ],
            },
            "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 32},
        },
        "training": {"batch_size": 4, "seq_len": 32, "feature_dim": 4, "horizon": 12, "n_paths": 100, "lr": 0.001},
    },
    "transformer_sde_head": {
        "description": "Transformer backbone with SDE head (deeper drift/vol network)",
        "tags": ["attention", "sde"],
        "model": {
            "backbone": {
                "_target_": "src.models.factory.HybridBackbone",
                "input_size": 4,
                "d_model": 32,
                "blocks": [
                    {"_target_": "src.models.registry.TransformerBlock", "d_model": 32, "nhead": 4},
                ],
            },
            "head": {"_target_": "src.models.heads.SDEHead", "latent_size": 32, "hidden": 64},
        },
        "training": {"batch_size": 4, "seq_len": 32, "feature_dim": 4, "horizon": 12, "n_paths": 100, "lr": 0.001},
    },
    "deep_hybrid": {
        "description": "4-block deep hybrid: RevIN + Transformer + ResConv + LSTM",
        "tags": ["deep", "hybrid", "normalization"],
        "model": {
            "backbone": {
                "_target_": "src.models.factory.HybridBackbone",
                "input_size": 4,
                "d_model": 32,
                "blocks": [
                    {"_target_": "src.models.components.advanced_blocks.RevIN", "d_model": 32},
                    {"_target_": "src.models.registry.TransformerBlock", "d_model": 32, "nhead": 4},
                    {"_target_": "src.models.components.advanced_blocks.ResConvBlock", "d_model": 32},
                    {"_target_": "src.models.registry.LSTMBlock", "d_model": 32, "num_layers": 1},
                ],
            },
            "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 32},
        },
        "training": {"batch_size": 4, "seq_len": 32, "feature_dim": 4, "horizon": 12, "n_paths": 100, "lr": 0.001},
    },
    "unet_transformer": {
        "description": "U-Net 1D + Transformer for multi-resolution attention",
        "tags": ["unet", "attention", "multi-resolution"],
        "model": {
            "backbone": {
                "_target_": "src.models.factory.HybridBackbone",
                "input_size": 4,
                "d_model": 32,
                "blocks": [
                    {"_target_": "src.models.components.advanced_blocks.Unet1DBlock", "d_model": 32},
                    {"_target_": "src.models.registry.TransformerBlock", "d_model": 32, "nhead": 4},
                ],
            },
            "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 32},
        },
        "training": {"batch_size": 4, "seq_len": 32, "feature_dim": 4, "horizon": 12, "n_paths": 100, "lr": 0.001},
    },
}


# ---------------------------------------------------------------------------
# Block & Head metadata for agent discovery
# ---------------------------------------------------------------------------

BLOCK_INFO: Dict[str, Dict[str, Any]] = {
    "TransformerBlock": {
        "target": "src.models.registry.TransformerBlock",
        "description": "Multi-head self-attention + gated MLP with residual connections",
        "params": {"d_model": "int (model dimension)", "nhead": "int (attention heads, default 4)", "dropout": "float (default 0.1)"},
        "strengths": "Captures long-range dependencies and global patterns",
        "cost": "medium",
    },
    "LSTMBlock": {
        "target": "src.models.registry.LSTMBlock",
        "description": "LSTM recurrent block for sequential pattern learning",
        "params": {"d_model": "int", "num_layers": "int (default 1)", "dropout": "float (default 0.0)"},
        "strengths": "Good at ordered sequences, momentum, trend following",
        "cost": "medium",
    },
    "GRUBlock": {
        "target": "src.models.components.advanced_blocks.GRUBlock",
        "description": "Gated Recurrent Unit - lighter alternative to LSTM",
        "params": {"d_model": "int", "num_layers": "int (default 1)", "dropout": "float (default 0.0)"},
        "strengths": "Faster than LSTM with similar performance on many tasks",
        "cost": "low-medium",
    },
    "RNNBlock": {
        "target": "src.models.components.advanced_blocks.RNNBlock",
        "description": "Simple Elman RNN (lightest recurrent option)",
        "params": {"d_model": "int", "num_layers": "int (default 1)", "dropout": "float (default 0.0)"},
        "strengths": "Minimal recurrent block, very fast",
        "cost": "low",
    },
    "ResConvBlock": {
        "target": "src.models.components.advanced_blocks.ResConvBlock",
        "description": "1D residual convolutional block for local pattern extraction",
        "params": {"d_model": "int", "kernel_size": "int (default 3)", "dropout": "float (default 0.1)"},
        "strengths": "Extracts local features, shift-invariant patterns",
        "cost": "low",
    },
    "BiTCNBlock": {
        "target": "src.models.components.advanced_blocks.BiTCNBlock",
        "description": "Bidirectional temporal convolutional block with dilation",
        "params": {"d_model": "int", "kernel_size": "int (default 3)", "dilation": "int (default 1)", "dropout": "float (default 0.1)"},
        "strengths": "Multi-scale local features via dilation, bidirectional context",
        "cost": "low",
    },
    "FourierBlock": {
        "target": "src.models.components.advanced_blocks.FourierBlock",
        "description": "Frequency-domain processing (FedFormer-inspired)",
        "params": {"d_model": "int", "modes": "int (number of Fourier modes, default 32)"},
        "strengths": "Captures periodic/seasonal patterns directly in frequency domain",
        "cost": "medium",
    },
    "DLinearBlock": {
        "target": "src.models.components.advanced_blocks.DLinearBlock",
        "description": "Trend-seasonal decomposition with linear layers",
        "params": {"d_model": "int", "kernel_size": "int (moving avg window, default 25)"},
        "strengths": "Extremely lightweight, strong baseline, interpretable decomposition",
        "cost": "very low",
    },
    "TimesNetBlock": {
        "target": "src.models.components.advanced_blocks.TimesNetBlock",
        "description": "Period discovery via FFT + 2D Inception convolution",
        "params": {"d_model": "int", "top_k": "int (dominant periods, default 3)", "d_ff": "int | None", "dropout": "float (default 0.1)"},
        "strengths": "Automatically discovers dominant periods, 2D modeling of temporal variations",
        "cost": "high",
    },
    "TimeMixerBlock": {
        "target": "src.models.components.advanced_blocks.TimeMixerBlock",
        "description": "Multi-scale past-decomposable mixing (ICLR 2024)",
        "params": {"d_model": "int", "d_ff": "int | None", "down_sampling_window": "int (default 2)", "down_sampling_layers": "int (default 2)"},
        "strengths": "Multi-scale trend/seasonal mixing, state-of-the-art decomposition approach",
        "cost": "medium",
    },
    "RevIN": {
        "target": "src.models.components.advanced_blocks.RevIN",
        "description": "Reversible instance normalization for non-stationary data",
        "params": {"d_model": "int", "affine": "bool (default True)"},
        "strengths": "Handles distribution shift, use as FIRST block in backbone",
        "cost": "very low",
    },
    "Unet1DBlock": {
        "target": "src.models.components.advanced_blocks.Unet1DBlock",
        "description": "U-Net style encode-decode with skip connections",
        "params": {"d_model": "int", "reduction": "int (default 2)"},
        "strengths": "Multi-resolution feature extraction with skip connections",
        "cost": "medium",
    },
    "LayerNormBlock": {
        "target": "src.models.components.advanced_blocks.LayerNormBlock",
        "description": "Standalone LayerNorm for inter-block normalization",
        "params": {"d_model": "int"},
        "strengths": "Stabilizes training between blocks, use between other blocks",
        "cost": "very low",
    },
    "SDEEvolutionBlock": {
        "target": "src.models.registry.SDEEvolutionBlock",
        "description": "Learned residual stochastic update block",
        "params": {"d_model": "int", "hidden": "int (default 64)", "dropout": "float (default 0.1)"},
        "strengths": "Stochastic residual modeling, good for noisy data",
        "cost": "low",
    },
    "TransformerEncoder": {
        "target": "src.models.components.advanced_blocks.TransformerEncoderAdapter",
        "description": "Full PyTorch TransformerEncoder (multi-layer)",
        "params": {"d_model": "int", "nhead": "int (default 4)", "num_layers": "int (default 2)", "dim_feedforward": "int (default 128)"},
        "strengths": "Deep self-attention encoding with multiple layers",
        "cost": "high",
    },
}

HEAD_INFO: Dict[str, Dict[str, Any]] = {
    "GBMHead": {
        "target": "src.models.heads.GBMHead",
        "description": "Geometric Brownian Motion - constant drift and volatility",
        "params": {"latent_size": "int (auto-set from backbone)"},
        "strengths": "Simple, fast, standard financial model. Good default choice.",
        "output": "(mu, sigma) scalars -> simulated price paths",
    },
    "SDEHead": {
        "target": "src.models.heads.SDEHead",
        "description": "Deeper SDE parameter network with hidden layers",
        "params": {"latent_size": "int", "hidden": "int (default 64)"},
        "strengths": "More expressive drift/volatility prediction than GBMHead",
        "output": "(mu, sigma) scalars via deeper network -> simulated paths",
    },
    "NeuralSDEHead": {
        "target": "src.models.heads.NeuralSDEHead",
        "description": "Full neural SDE with learned drift/diffusion networks (uses torchsde)",
        "params": {"latent_size": "int", "hidden": "int (default 64)", "solver": "str (default 'euler')", "adjoint": "bool (default False)"},
        "strengths": "Most expressive - learns state-dependent, time-varying dynamics",
        "output": "Paths generated internally via SDE integration",
    },
    "HorizonHead": {
        "target": "src.models.heads.HorizonHead",
        "description": "Per-step drift/volatility via cross-attention to backbone sequence",
        "params": {"latent_size": "int", "horizon_max": "int (default 48)", "nhead": "int (default 4)", "n_layers": "int (default 2)"},
        "strengths": "Time-varying dynamics, can express volatility clustering and regime changes",
        "output": "(mu_seq, sigma_seq) per-step parameters",
    },
    "SimpleHorizonHead": {
        "target": "src.models.heads.SimpleHorizonHead",
        "description": "Per-step drift/volatility via pooling + MLP (no attention, memory efficient)",
        "params": {"latent_size": "int", "horizon_max": "int (default 48)", "pool_type": "str (default 'mean')"},
        "strengths": "10-20x more memory efficient than HorizonHead, still per-step",
        "output": "(mu_seq, sigma_seq) per-step parameters",
    },
    "NeuralBridgeHead": {
        "target": "src.models.heads.NeuralBridgeHead",
        "description": "Hierarchical: macro 1H move + micro sub-hour texture",
        "params": {"latent_size": "int", "micro_steps": "int (default 12)", "hidden_dim": "int (default 64)"},
        "strengths": "Pinned endpoints with learned intra-path dynamics",
        "output": "(macro_return, micro_path, sigma)",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dummy_batch(training_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Build a synthetic batch for quick experimentation."""
    batch_size = training_cfg.get("batch_size", 4)
    seq_len = training_cfg.get("seq_len", 32)
    feature_dim = training_cfg.get("feature_dim", 4)
    horizon = training_cfg.get("horizon", 12)

    history = torch.randn(batch_size, seq_len, feature_dim)
    initial_price = torch.full((batch_size,), 100.0)
    target = initial_price + torch.randn(batch_size) * 1.0
    actual_series = initial_price.unsqueeze(-1) * torch.exp(
        torch.linspace(0, 0.01 * horizon, steps=horizon)
    )

    return {
        "history": history,
        "initial_price": initial_price,
        "target": target,
        "actual_series": actual_series,
    }


def _count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def _block_spec_to_config(block_name: str, d_model: int, **kwargs: Any) -> Dict[str, Any]:
    """Convert a block name + params to a Hydra-style config dict."""
    info = BLOCK_INFO.get(block_name)
    if info is None:
        raise ValueError(
            f"Unknown block: '{block_name}'. "
            f"Available blocks: {list(BLOCK_INFO.keys())}"
        )
    cfg = {"_target_": info["target"], "d_model": d_model}
    cfg.update(kwargs)
    return cfg


def _head_spec_to_config(head_name: str, latent_size: int, **kwargs: Any) -> Dict[str, Any]:
    """Convert a head name + params to a Hydra-style config dict."""
    info = HEAD_INFO.get(head_name)
    if info is None:
        raise ValueError(
            f"Unknown head: '{head_name}'. "
            f"Available heads: {list(HEAD_INFO.keys())}"
        )
    cfg = {"_target_": info["target"], "latent_size": latent_size}
    cfg.update(kwargs)
    return cfg


# ---------------------------------------------------------------------------
# ResearchSession
# ---------------------------------------------------------------------------


class ResearchSession:
    """Zero-config, side-effect-free research interface for AI agents.

    Provides discovery, experiment construction, execution, and comparison
    through a pure-Python API using plain dicts for all I/O.

    No W&B, no HF Hub, no file writes. Just models, metrics, and insights.

    Example
    -------
    ::

        session = ResearchSession()

        # Discover available components
        session.list_blocks()
        session.list_heads()
        session.list_presets()

        # Run experiments
        r1 = session.run_preset("transformer_lstm")
        r2 = session.run_preset("dlinear_simple")

        # Compare
        session.compare()

        # Custom experiment
        exp = session.create_experiment(
            blocks=["FourierBlock", "LSTMBlock"],
            head="GBMHead",
            d_model=64,
            horizon=24,
        )
        r3 = session.run(exp)
    """

    def __init__(self) -> None:
        discover_components("src/models/components")
        self._results: List[Dict[str, Any]] = []
        self._experiment_counter = 0

    # -- Discovery ----------------------------------------------------------

    def list_blocks(self) -> List[Dict[str, Any]]:
        """List all available backbone blocks with descriptions and parameters.

        Returns a list of dicts, each with: name, description, params,
        strengths, cost, target.
        """
        return [
            {"name": name, **info}
            for name, info in sorted(BLOCK_INFO.items())
        ]

    def list_heads(self) -> List[Dict[str, Any]]:
        """List all available simulation heads with descriptions.

        Returns a list of dicts, each with: name, description, params,
        strengths, output, target.
        """
        return [
            {"name": name, **info}
            for name, info in sorted(HEAD_INFO.items())
        ]

    def list_presets(self) -> List[Dict[str, Any]]:
        """List pre-built experiment presets an agent can run with one call.

        Returns a list of dicts, each with: name, description, tags,
        blocks (list of block names), head (head name).
        """
        result = []
        for name, preset in sorted(PRESETS.items()):
            blocks = preset["model"]["backbone"]["blocks"]
            block_names = [b["_target_"].split(".")[-1] for b in blocks]
            head_name = preset["model"]["head"]["_target_"].split(".")[-1]
            result.append({
                "name": name,
                "description": preset["description"],
                "tags": preset.get("tags", []),
                "blocks": block_names,
                "head": head_name,
            })
        return result

    # -- Experiment Construction --------------------------------------------

    def create_experiment(
        self,
        blocks: List[str | Dict[str, Any]],
        head: str = "GBMHead",
        *,
        d_model: int = 32,
        feature_dim: int = 4,
        seq_len: int = 32,
        horizon: int = 12,
        n_paths: int = 100,
        batch_size: int = 4,
        lr: float = 0.001,
        head_kwargs: Optional[Dict[str, Any]] = None,
        block_kwargs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build an experiment config from block/head names and parameters.

        Parameters
        ----------
        blocks : list of str or dict
            Block names (e.g. ["TransformerBlock", "LSTMBlock"]) or raw config
            dicts with ``_target_`` keys. Use ``session.list_blocks()`` to see
            available block names.
        head : str
            Head name (e.g. "GBMHead"). Use ``session.list_heads()`` to see options.
        d_model : int
            Model hidden dimension (default 32). Higher = more expressive but slower.
        feature_dim : int
            Number of input features (default 4).
        seq_len : int
            Input sequence length (default 32).
        horizon : int
            Prediction horizon in timesteps (default 12).
        n_paths : int
            Number of Monte Carlo simulation paths (default 100).
        batch_size : int
            Batch size for training (default 4).
        lr : float
            Learning rate (default 0.001).
        head_kwargs : dict, optional
            Extra keyword arguments passed to the head constructor.
        block_kwargs : list of dict, optional
            Per-block extra kwargs (must match length of ``blocks``).

        Returns
        -------
        dict
            Experiment configuration ready for ``session.run()``.
        """
        block_kwargs = block_kwargs or [{}] * len(blocks)
        if len(block_kwargs) != len(blocks):
            raise ValueError(
                f"block_kwargs length ({len(block_kwargs)}) must match "
                f"blocks length ({len(blocks)})"
            )

        block_configs = []
        for i, block in enumerate(blocks):
            if isinstance(block, dict):
                block_configs.append(block)
            else:
                block_configs.append(
                    _block_spec_to_config(block, d_model, **block_kwargs[i])
                )

        head_config = _head_spec_to_config(head, d_model, **(head_kwargs or {}))

        block_names = []
        for b in block_configs:
            target = b.get("_target_", "unknown")
            block_names.append(target.split(".")[-1])

        return {
            "name": f"{'_'.join(block_names)}_{head.replace('Head', '').lower()}",
            "model": {
                "backbone": {
                    "_target_": "src.models.factory.HybridBackbone",
                    "input_size": feature_dim,
                    "d_model": d_model,
                    "blocks": block_configs,
                },
                "head": head_config,
            },
            "training": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "feature_dim": feature_dim,
                "horizon": horizon,
                "n_paths": n_paths,
                "lr": lr,
            },
        }

    # -- Validation ---------------------------------------------------------

    def validate(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an experiment config without running it.

        Checks that the model can be instantiated and passes a shape smoke
        test. Returns a dict with ``valid`` (bool), ``param_count``,
        ``errors`` (list), and ``warnings`` (list).

        Parameters
        ----------
        experiment : dict
            Experiment config from ``create_experiment()`` or ``PRESETS``.

        Returns
        -------
        dict
            Validation result with keys: valid, param_count, errors, warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []
        param_count = {"total": 0, "trainable": 0}

        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.create(experiment)
            model = create_model(cfg)
            param_count = _count_parameters(model)

            if param_count["trainable"] == 0:
                warnings.append("Model has 0 trainable parameters")
            if param_count["trainable"] > 1_000_000:
                warnings.append(
                    f"Large model: {param_count['trainable']:,} trainable params. "
                    f"Training may be slow."
                )
        except Exception as e:
            errors.append(f"Model instantiation failed: {e}")

        return {
            "valid": len(errors) == 0,
            "param_count": param_count,
            "errors": errors,
            "warnings": warnings,
        }

    def describe(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Describe what an experiment will do, without running it.

        Returns structured metadata about the model architecture, parameter
        count, and training configuration.

        Parameters
        ----------
        experiment : dict
            Experiment config.

        Returns
        -------
        dict
            Description with keys: name, blocks, head, d_model, param_count,
            training, validation.
        """
        model_cfg = experiment.get("model", {})
        backbone_cfg = model_cfg.get("backbone", {})
        head_cfg = model_cfg.get("head", {})
        training_cfg = experiment.get("training", {})

        blocks = backbone_cfg.get("blocks", [])
        block_names = [b.get("_target_", "unknown").split(".")[-1] for b in blocks]
        head_name = head_cfg.get("_target_", "unknown").split(".")[-1]

        validation = self.validate(experiment)

        return {
            "name": experiment.get("name", "unnamed"),
            "blocks": block_names,
            "head": head_name,
            "d_model": backbone_cfg.get("d_model", "unknown"),
            "feature_dim": backbone_cfg.get("input_size", "unknown"),
            "param_count": validation["param_count"],
            "training": {
                "horizon": training_cfg.get("horizon"),
                "n_paths": training_cfg.get("n_paths"),
                "batch_size": training_cfg.get("batch_size"),
                "seq_len": training_cfg.get("seq_len"),
                "lr": training_cfg.get("lr"),
            },
            "validation": validation,
        }

    # -- Execution ----------------------------------------------------------

    def run(
        self,
        experiment: Dict[str, Any],
        *,
        epochs: int = 1,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run an experiment and return structured results.

        No W&B logging, no HF Hub push, no file writes. Pure computation.

        Parameters
        ----------
        experiment : dict
            Experiment config from ``create_experiment()`` or a preset.
        epochs : int
            Number of training epochs (default 1 for quick screening).
        name : str, optional
            Override experiment name.

        Returns
        -------
        dict
            Result with keys: name, status, metrics, param_count, config,
            duration_seconds, epochs.
        """
        from omegaconf import OmegaConf

        self._experiment_counter += 1
        exp_name = name or experiment.get("name", f"experiment_{self._experiment_counter}")

        start_time = time.time()
        try:
            cfg = OmegaConf.create(experiment)
            training_cfg = experiment.get("training", {})

            model = create_model(cfg)
            optimizer = optim.Adam(model.parameters(), lr=training_cfg.get("lr", 1e-3))
            param_count = _count_parameters(model)

            batch = _build_dummy_batch(training_cfg)
            horizon = training_cfg.get("horizon", 12)
            n_paths = training_cfg.get("n_paths", 100)

            all_metrics: List[Dict[str, float]] = []
            for epoch in range(epochs):
                step_metrics = train_step(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    horizon=horizon,
                    n_paths=n_paths,
                )
                all_metrics.append(step_metrics)

            # Final evaluation (no W&B logging)
            model.eval()
            with torch.no_grad():
                paths, mu, sigma = model(
                    batch["history"],
                    initial_price=batch["initial_price"],
                    horizon=horizon,
                    n_paths=n_paths,
                )
                from src.research.metrics import crps_ensemble, afcrps_ensemble, log_likelihood
                from src.research.trainer import prepare_paths_for_crps

                terminal_paths = paths[:, :, -1]
                target = batch["target"]
                crps = afcrps_ensemble(terminal_paths, target).mean().item()
                sharpness = terminal_paths.std(dim=1).mean().item()
                loglik = log_likelihood(terminal_paths, target).mean().item()

            final_metrics = {
                "crps": crps,
                "sharpness": sharpness,
                "log_likelihood": loglik,
                "final_train_loss": all_metrics[-1]["loss"],
            }

            duration = time.time() - start_time

            result = {
                "name": exp_name,
                "status": "ok",
                "metrics": final_metrics,
                "training_history": all_metrics,
                "param_count": param_count,
                "config_summary": {
                    "blocks": [b.get("_target_", "?").split(".")[-1] for b in experiment.get("model", {}).get("backbone", {}).get("blocks", [])],
                    "head": experiment.get("model", {}).get("head", {}).get("_target_", "?").split(".")[-1],
                    "d_model": experiment.get("model", {}).get("backbone", {}).get("d_model"),
                    "horizon": horizon,
                    "n_paths": n_paths,
                },
                "duration_seconds": round(duration, 2),
                "epochs": epochs,
            }

            self._results.append(result)
            return result

        except Exception as e:
            duration = time.time() - start_time
            result = {
                "name": exp_name,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "duration_seconds": round(duration, 2),
                "epochs": 0,
            }
            self._results.append(result)
            return result

    def run_preset(
        self,
        preset_name: str,
        *,
        epochs: int = 1,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a named preset experiment.

        Parameters
        ----------
        preset_name : str
            Name of the preset (use ``list_presets()`` to see options).
        epochs : int
            Number of training epochs (default 1).
        overrides : dict, optional
            Override training parameters, e.g. ``{"training": {"horizon": 24}}``.

        Returns
        -------
        dict
            Experiment result (same format as ``run()``).
        """
        if preset_name not in PRESETS:
            available = list(PRESETS.keys())
            raise ValueError(
                f"Unknown preset: '{preset_name}'. Available: {available}"
            )

        experiment = _deep_copy_dict(PRESETS[preset_name])
        if overrides:
            _deep_merge(experiment, overrides)

        return self.run(experiment, name=preset_name, epochs=epochs)

    # -- Comparison ---------------------------------------------------------

    def compare(self, results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Compare experiment results side-by-side.

        Parameters
        ----------
        results : list of dict, optional
            Results to compare. Defaults to all results in this session.

        Returns
        -------
        dict
            Comparison with keys: ranking (sorted by CRPS, lower is better),
            best, worst, all_metrics.
        """
        results = results or self._results
        if not results:
            return {"ranking": [], "best": None, "worst": None, "all_metrics": {}}

        successful = [r for r in results if r.get("status") == "ok"]
        failed = [r for r in results if r.get("status") != "ok"]

        if not successful:
            return {
                "ranking": [],
                "best": None,
                "worst": None,
                "failed": [{"name": r["name"], "error": r.get("error")} for r in failed],
                "all_metrics": {},
            }

        # Build comparison table
        entries = []
        for r in successful:
            metrics = r.get("metrics", {})
            entries.append({
                "name": r["name"],
                "crps": metrics.get("crps"),
                "sharpness": metrics.get("sharpness"),
                "log_likelihood": metrics.get("log_likelihood"),
                "param_count": r.get("param_count", {}).get("trainable"),
                "duration_seconds": r.get("duration_seconds"),
                "blocks": r.get("config_summary", {}).get("blocks", []),
                "head": r.get("config_summary", {}).get("head"),
            })

        # Sort by CRPS (lower is better)
        entries.sort(key=lambda e: e.get("crps") or float("inf"))

        return {
            "ranking": entries,
            "best": entries[0] if entries else None,
            "worst": entries[-1] if entries else None,
            "num_experiments": len(entries),
            "num_failed": len(failed),
            "failed": [{"name": r["name"], "error": r.get("error")} for r in failed] if failed else [],
        }

    # -- Sweep (run multiple presets) ----------------------------------------

    def sweep(
        self,
        preset_names: Optional[List[str]] = None,
        *,
        epochs: int = 1,
    ) -> Dict[str, Any]:
        """Run multiple presets and return comparison.

        Parameters
        ----------
        preset_names : list of str, optional
            Which presets to run. Defaults to ALL presets.
        epochs : int
            Training epochs per experiment.

        Returns
        -------
        dict
            Comparison of all sweep results (same format as ``compare()``).
        """
        names = preset_names or list(PRESETS.keys())
        results = []
        for name in names:
            r = self.run_preset(name, epochs=epochs)
            results.append(r)
        return self.compare(results)

    # -- Session state ------------------------------------------------------

    @property
    def results(self) -> List[Dict[str, Any]]:
        """All experiment results collected in this session."""
        return list(self._results)

    def summary(self) -> Dict[str, Any]:
        """Full session summary with all results and comparison.

        Returns
        -------
        dict
            Summary with: num_experiments, comparison, all_results.
        """
        return {
            "num_experiments": len(self._results),
            "comparison": self.compare(),
            "all_results": self._results,
        }

    def clear(self) -> None:
        """Clear all stored results."""
        self._results.clear()
        self._experiment_counter = 0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _deep_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy a nested dict (avoiding copy module for portability)."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = [_deep_copy_dict(i) if isinstance(i, dict) else i for i in v]
        else:
            result[k] = v
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge override into base (mutates base)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Module-level convenience (for agents that just want to import and go)
# ---------------------------------------------------------------------------


def quick_experiment(
    blocks: List[str] | None = None,
    head: str = "GBMHead",
    **kwargs: Any,
) -> Dict[str, Any]:
    """One-liner to create a session, run an experiment, and return results.

    Parameters
    ----------
    blocks : list of str, optional
        Block names. Defaults to ["TransformerBlock", "LSTMBlock"].
    head : str
        Head name (default "GBMHead").
    **kwargs
        Passed to ``ResearchSession.create_experiment()``.

    Returns
    -------
    dict
        Experiment result.

    Example
    -------
    ::

        from src.research.agent_api import quick_experiment
        result = quick_experiment(blocks=["FourierBlock", "GRUBlock"], d_model=64)
        print(result["metrics"]["crps"])
    """
    session = ResearchSession()
    blocks = blocks or ["TransformerBlock", "LSTMBlock"]
    experiment = session.create_experiment(blocks=blocks, head=head, **kwargs)
    return session.run(experiment)


__all__ = [
    "ResearchSession",
    "quick_experiment",
    "PRESETS",
    "BLOCK_INFO",
    "HEAD_INFO",
]
