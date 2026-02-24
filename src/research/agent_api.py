"""High-level research session API for agent-driven experimentation.

Provides :class:`ResearchSession` — a stateful, zero-config entry point that
wraps the registry, model factory, trainer, and metrics into a single object
suitable for interactive or automated research workflows.

Also exports :func:`quick_experiment` as a convenience one-liner.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim

from osa.models.factory import (
    HEAD_REGISTRY,
    HybridBackbone,
    SynthModel,
    build_model,
    _smoke_test_model,
)
from osa.models.registry import discover_components, registry
from osa.research.trainer import DataToModelAdapter, Trainer, prepare_paths_for_crps
from osa.research.metrics import crps_ensemble, log_likelihood


# ── Block cost heuristics (relative) ────────────────────────────────────
_BLOCK_COST: Dict[str, str] = {
    "transformerblock": "medium",
    "lstmblock": "medium",
    "sdeevolutionblock": "low",
    "rnnblock": "low",
    "grublock": "medium",
    "resconvblock": "low",
    "bitcnblock": "low",
    "patchembedding": "low",
    "unet1dblock": "medium",
    "transformerencoder": "high",
    "transformerdecoder": "high",
    "fourierblock": "medium",
    "laststepadapter": "low",
    "revin": "low",
    "flexiblepatchembed": "low",
    "channelrejoin": "low",
    "multiscalepatcher": "medium",
    "dlinearblock": "low",
    "layernormblock": "low",
    "timesnetblock": "high",
    "timemixerblock": "high",
    "patchmixerblock": "medium",
}

_BLOCK_BEST_FOR: Dict[str, str] = {
    "transformerblock": "general time-series with attention",
    "lstmblock": "sequential dependencies",
    "sdeevolutionblock": "residual stochastic dynamics",
    "rnnblock": "simple sequential modeling",
    "grublock": "sequential dependencies (lighter than LSTM)",
    "resconvblock": "local pattern extraction",
    "bitcnblock": "bidirectional temporal patterns",
    "patchembedding": "downsampling long sequences via patches",
    "unet1dblock": "multi-resolution feature extraction",
    "transformerencoder": "deep attention stacks",
    "transformerdecoder": "autoregressive decoding",
    "fourierblock": "frequency-domain filtering",
    "laststepadapter": "pooling before heads",
    "revin": "non-stationary input normalization",
    "flexiblepatchembed": "channel-independent patching",
    "channelrejoin": "merging channel-independent branches",
    "multiscalepatcher": "multi-resolution patching",
    "dlinearblock": "trend-seasonal decomposition",
    "layernormblock": "inter-block normalization",
    "timesnetblock": "periodic pattern discovery (FFT + 2D conv)",
    "timemixerblock": "multi-scale trend/seasonal mixing",
    "patchmixerblock": "MLP-Mixer style token/feature mixing",
}

# ── Head metadata ────────────────────────────────────────────────────────
_HEAD_META: Dict[str, Dict[str, str]] = {
    "gbm": {
        "expressiveness": "low",
        "description": "Geometric Brownian Motion — constant drift and volatility",
    },
    "sde": {
        "expressiveness": "medium",
        "description": "MLP-based SDE parameters — richer mu/sigma mapping",
    },
    "neural_sde": {
        "expressiveness": "very high",
        "description": "Neural SDE with learned drift/diffusion networks (requires torchsde)",
    },
    "horizon": {
        "expressiveness": "high",
        "description": "Cross-attention per-step mu/sigma over the full backbone sequence",
    },
    "simple_horizon": {
        "expressiveness": "medium",
        "description": "Pooling-based per-step mu/sigma (memory-efficient HorizonHead)",
    },
    "mixture_density": {
        "expressiveness": "high",
        "description": "K-component Gaussian mixture for fat tails and multimodality",
    },
    "vol_term_structure": {
        "expressiveness": "medium",
        "description": "Parametric volatility term structure (4 scalars → horizon curves)",
    },
    "neural_bridge": {
        "expressiveness": "high",
        "description": "Hierarchical macro + micro texture with bridge constraints",
    },
}

# ── Built-in presets ─────────────────────────────────────────────────────
_PRESETS: List[Dict[str, Any]] = [
    {
        "name": "baseline_gbm",
        "head": "gbm",
        "blocks": ["TransformerBlock", "LSTMBlock"],
        "tags": ["baseline", "simple"],
        "description": "Transformer + LSTM backbone with GBM head",
    },
    {
        "name": "sde_transformer",
        "head": "sde",
        "blocks": ["TransformerBlock"],
        "tags": ["sde", "attention"],
        "description": "Transformer backbone with SDE head",
    },
    {
        "name": "horizon_transformer",
        "head": "simple_horizon",
        "blocks": ["TransformerBlock", "LSTMBlock"],
        "tags": ["horizon", "per-step"],
        "description": "Transformer + LSTM with per-step SimpleHorizon head",
    },
    {
        "name": "mixture_density",
        "head": "mixture_density",
        "blocks": ["TransformerBlock"],
        "tags": ["mixture", "fat-tails"],
        "description": "Transformer with mixture density head for fat-tailed distributions",
    },
    {
        "name": "vol_term_structure",
        "head": "vol_term_structure",
        "blocks": ["TransformerBlock", "LSTMBlock"],
        "tags": ["vol-curve", "parametric"],
        "description": "Transformer + LSTM with parametric volatility term structure",
    },
    {
        "name": "revin_transformer",
        "head": "gbm",
        "blocks": ["RevIN", "TransformerBlock", "LSTMBlock"],
        "tags": ["revin", "normalization"],
        "description": "RevIN-normalized Transformer + LSTM with GBM head",
    },
    {
        "name": "dlinear_simple",
        "head": "gbm",
        "blocks": ["DLinearBlock"],
        "tags": ["linear", "lightweight"],
        "description": "DLinear decomposition with GBM head (no attention)",
    },
    {
        "name": "timesnet_sde",
        "head": "sde",
        "blocks": ["TimesNetBlock"],
        "tags": ["periodic", "fft"],
        "description": "TimesNet periodic block with SDE head",
    },
]


def _ensure_components_discovered() -> None:
    """Trigger auto-discovery if no blocks have been registered from components."""
    discover_components("src/models/components")


def _resolve_head_target(head_key: str) -> str:
    """Return the class name for a head registry key."""
    cls = HEAD_REGISTRY.get(head_key)
    if cls is None:
        raise KeyError(f"Unknown head '{head_key}'. Available: {list(HEAD_REGISTRY.keys())}")
    return cls.__name__


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _build_dummy_batch(
    batch_size: int,
    seq_len: int,
    feature_dim: int,
    horizon: int,
) -> Dict[str, torch.Tensor]:
    """Create a synthetic batch for validation / training."""
    history = torch.randn(batch_size, seq_len, feature_dim)
    initial_price = torch.ones(batch_size)
    target_factors = torch.exp(torch.randn(batch_size, horizon) * 0.01)
    return {
        "history": history,
        "initial_price": initial_price,
        "target_factors": target_factors,
    }


class ResearchSession:
    """Stateful research session for interactive or agent-driven experimentation.

    Wraps the block/head registries, model factory, trainer, and metrics into a
    single zero-argument constructor.  Results from :meth:`run`, :meth:`run_preset`,
    and :meth:`sweep` are accumulated in memory and can be ranked via :meth:`compare`.

    Parameters
    ----------
    (none — the constructor takes no arguments)

    Examples
    --------
    >>> session = ResearchSession()
    >>> exp = session.create_experiment(
    ...     blocks=["TransformerBlock"], head="gbm",
    ...     d_model=32, feature_dim=4, seq_len=32,
    ...     horizon=12, n_paths=100, batch_size=4, lr=0.001,
    ... )
    >>> result = session.run(exp, epochs=5, name="quick-test")
    >>> session.compare()
    """

    def __init__(self) -> None:
        _ensure_components_discovered()
        self._results: List[Dict[str, Any]] = []

    # ── Discovery methods ────────────────────────────────────────────────

    def list_blocks(self) -> List[Dict[str, Any]]:
        """Return all registered backbone blocks with metadata.

        Returns
        -------
        list[dict]
            Each dict has keys ``name``, ``cost``, ``best_for``, ``kind``,
            ``preserves_seq_len``, ``description``.
        """
        _ensure_components_discovered()
        blocks = registry.list_blocks(kind="block")
        return [
            {
                "name": info.name,
                "cost": _BLOCK_COST.get(info.name, "unknown"),
                "best_for": _BLOCK_BEST_FOR.get(info.name, ""),
                "kind": info.kind,
                "preserves_seq_len": info.preserves_seq_len,
                "description": info.description,
            }
            for info in blocks
        ]

    def list_heads(self) -> List[Dict[str, str]]:
        """Return all registered prediction heads with metadata.

        Returns
        -------
        list[dict]
            Each dict has keys ``name``, ``expressiveness``, ``description``.
        """
        return [
            {
                "name": key,
                "expressiveness": _HEAD_META.get(key, {}).get("expressiveness", "unknown"),
                "description": _HEAD_META.get(key, {}).get("description", ""),
            }
            for key in HEAD_REGISTRY
        ]

    def list_presets(self) -> List[Dict[str, Any]]:
        """Return built-in block+head preset combinations.

        Returns
        -------
        list[dict]
            Each dict has keys ``name``, ``head``, ``blocks``, ``tags``,
            ``description``.
        """
        return [dict(p) for p in _PRESETS]

    # ── Experiment construction ───────────────────────────────────────────

    def create_experiment(
        self,
        blocks: List[str],
        head: str,
        d_model: int,
        feature_dim: int,
        seq_len: int,
        horizon: int,
        n_paths: int,
        batch_size: int,
        lr: float,
        head_kwargs: Optional[Dict[str, Any]] = None,
        block_kwargs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build an experiment config dict from explicit parameters.

        Validates that all block and head names exist in the registries before
        returning the config.

        Parameters
        ----------
        blocks : list[str]
            Block names (case-insensitive) to stack in the backbone.
        head : str
            Head registry key (e.g. ``"gbm"``, ``"sde"``, ``"horizon"``).
        d_model : int
            Latent dimension for the backbone.
        feature_dim : int
            Input feature dimension.
        seq_len : int
            Sequence length fed to the backbone.
        horizon : int
            Prediction horizon (number of future steps).
        n_paths : int
            Number of Monte-Carlo simulation paths.
        batch_size : int
            Training batch size.
        lr : float
            Learning rate.
        head_kwargs : dict, optional
            Extra keyword arguments forwarded to the head constructor.
        block_kwargs : list[dict], optional
            Per-block extra keyword arguments (must match ``len(blocks)``).

        Returns
        -------
        dict
            Experiment config dict ready for :meth:`validate`, :meth:`describe`,
            or :meth:`run`.

        Raises
        ------
        KeyError
            If a block or head name is not found in the registries.
        ValueError
            If ``block_kwargs`` length doesn't match ``blocks``.
        """
        _ensure_components_discovered()

        # Validate block names
        for name in blocks:
            key = name.lower()
            if key not in registry.blocks and key not in registry.components:
                available = sorted(
                    list(registry.blocks.keys()) + list(registry.components.keys())
                )
                raise KeyError(
                    f"Unknown block '{name}'. Available: {available}"
                )

        # Validate head name
        head_lower = head.lower()
        # Accept both registry keys and class names
        head_key: Optional[str] = None
        for k, cls in HEAD_REGISTRY.items():
            if k == head_lower or cls.__name__.lower() == head_lower:
                head_key = k
                break
        if head_key is None:
            raise KeyError(
                f"Unknown head '{head}'. Available: {list(HEAD_REGISTRY.keys())}"
            )

        if block_kwargs is not None and len(block_kwargs) != len(blocks):
            raise ValueError(
                f"block_kwargs has {len(block_kwargs)} entries but blocks has "
                f"{len(blocks)}. They must match."
            )

        # Build head config
        head_cls_name = HEAD_REGISTRY[head_key].__name__
        head_cfg: Dict[str, Any] = {
            "_target_": f"osa.models.heads.{head_cls_name}",
            "latent_size": d_model,
        }
        if head_kwargs:
            head_cfg.update(head_kwargs)

        # Build backbone block list for Hydra
        hydra_blocks: List[Dict[str, Any]] = []
        for i, block_name in enumerate(blocks):
            block_entry: Dict[str, Any] = {
                "_target_": f"osa.models.registry.registry.{block_name}",
                "d_model": d_model,
            }
            if block_kwargs and block_kwargs[i]:
                block_entry.update(block_kwargs[i])
            hydra_blocks.append(block_entry)

        return {
            "model": {
                "backbone": {
                    "blocks": list(blocks),
                    "d_model": d_model,
                    "feature_dim": feature_dim,
                    "seq_len": seq_len,
                    "block_kwargs": block_kwargs if block_kwargs else [],
                    "_hydra_blocks": hydra_blocks,
                },
                "head": head_cfg,
            },
            "training": {
                "horizon": horizon,
                "n_paths": n_paths,
                "batch_size": batch_size,
                "lr": lr,
            },
        }

    # ── Validation ────────────────────────────────────────────────────────

    def _build_model_from_experiment(self, exp: Dict[str, Any]) -> SynthModel:
        """Instantiate a SynthModel from an experiment config dict."""
        _ensure_components_discovered()

        model_cfg = exp["model"]
        backbone_cfg = model_cfg["backbone"]
        head_cfg = model_cfg["head"]

        d_model = backbone_cfg["d_model"]
        feature_dim = backbone_cfg.get("feature_dim", 4)

        # Resolve block classes
        block_names = backbone_cfg["blocks"]
        block_kwargs_list = backbone_cfg.get("block_kwargs", [])
        block_instances = []
        for i, name in enumerate(block_names):
            key = name.lower()
            if key in registry.blocks:
                cls = registry.blocks[key]
            elif key in registry.components:
                cls = registry.components[key]
            else:
                raise KeyError(f"Block '{name}' not found in registry")

            kwargs: Dict[str, Any] = {"d_model": d_model}
            if block_kwargs_list and i < len(block_kwargs_list) and block_kwargs_list[i]:
                kwargs.update(block_kwargs_list[i])
            block_instances.append(cls(**kwargs))

        backbone = HybridBackbone(
            input_size=feature_dim,
            d_model=d_model,
            blocks=block_instances,
            validate_shapes=True,
            strict_shapes=False,
        )

        # Resolve head
        head_target = head_cfg.get("_target_", "")
        head_cls_name = head_target.rsplit(".", 1)[-1] if head_target else ""

        head_cls = None
        for k, cls in HEAD_REGISTRY.items():
            if cls.__name__ == head_cls_name:
                head_cls = cls
                break
        if head_cls is None:
            raise KeyError(f"Head class '{head_cls_name}' not found in HEAD_REGISTRY")

        # Build head kwargs (exclude _target_)
        head_kwargs_resolved = {
            k: v for k, v in head_cfg.items() if k != "_target_"
        }
        # Ensure latent_size matches backbone output
        latent_size = getattr(backbone, "output_dim", None) or d_model
        head_kwargs_resolved["latent_size"] = latent_size

        head = head_cls(**head_kwargs_resolved)
        return SynthModel(backbone=backbone, head=head)

    def validate(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an experiment config without training.

        Builds the model graph, counts parameters, and reports any issues.

        Parameters
        ----------
        exp : dict
            Experiment config from :meth:`create_experiment`.

        Returns
        -------
        dict
            ``{"valid": bool, "errors": list, "warnings": list, "param_count": int}``
        """
        errors: List[str] = []
        warnings: List[str] = []
        param_count = 0

        try:
            model = self._build_model_from_experiment(exp)
            param_count = _count_params(model)

            # Smoke test
            feature_dim = exp["model"]["backbone"].get("feature_dim", 4)
            seq_len = exp["model"]["backbone"].get("seq_len", 32)
            _smoke_test_model(model, feature_dim, seq_len)
        except Exception as e:
            errors.append(str(e))

        # Warnings for unusual configs
        training = exp.get("training", {})
        if training.get("lr", 0) > 0.01:
            warnings.append("Learning rate > 0.01 may cause instability")
        if training.get("n_paths", 0) < 10:
            warnings.append("Very few paths (<10) may give noisy CRPS estimates")
        if param_count > 1_000_000:
            warnings.append(f"Large model ({param_count:,} params) — may be slow to train")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "param_count": param_count,
        }

    def describe(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """Return a full description of the experiment config.

        Includes block details, head info, parameter count, training config,
        and validation results.

        Parameters
        ----------
        exp : dict
            Experiment config from :meth:`create_experiment`.

        Returns
        -------
        dict
            Description dict with keys ``blocks``, ``head``, ``param_count``,
            ``training``, ``validation``.
        """
        validation = self.validate(exp)
        model_cfg = exp.get("model", {})
        backbone_cfg = model_cfg.get("backbone", {})
        head_cfg = model_cfg.get("head", {})
        training_cfg = exp.get("training", {})

        block_names = backbone_cfg.get("blocks", [])
        block_details = []
        for name in block_names:
            key = name.lower()
            try:
                info = registry.get_info(key)
                block_details.append({
                    "name": info.name,
                    "description": info.description,
                    "preserves_seq_len": info.preserves_seq_len,
                })
            except KeyError:
                block_details.append({"name": name, "description": "unknown", "preserves_seq_len": None})

        # Head info
        head_target = head_cfg.get("_target_", "")
        head_cls_name = head_target.rsplit(".", 1)[-1] if head_target else "unknown"
        head_key = None
        for k, cls in HEAD_REGISTRY.items():
            if cls.__name__ == head_cls_name:
                head_key = k
                break

        head_info = {
            "name": head_key or head_cls_name,
            "class": head_cls_name,
            **(
                _HEAD_META.get(head_key, {})
                if head_key
                else {}
            ),
        }

        return {
            "blocks": block_details,
            "head": head_info,
            "backbone": {
                "d_model": backbone_cfg.get("d_model"),
                "feature_dim": backbone_cfg.get("feature_dim"),
                "seq_len": backbone_cfg.get("seq_len"),
            },
            "param_count": validation["param_count"],
            "training": dict(training_cfg),
            "validation": validation,
        }

    # ── Execution ─────────────────────────────────────────────────────────

    def run(
        self,
        exp: Dict[str, Any],
        epochs: int = 1,
        name: Optional[str] = None,
        data_loader: Any = None,
    ) -> Dict[str, Any]:
        """Train a model from an experiment config and return metrics.

        Results are accumulated in the session for later :meth:`compare`.
        Execution errors are returned in the result dict — this method never
        raises on training failures.

        Parameters
        ----------
        exp : dict
            Experiment config from :meth:`create_experiment`.
        epochs : int
            Number of training epochs (default 1).
        name : str, optional
            Human-readable name for this run.
        data_loader : optional
            An iterable of batches.  When *None*, synthetic data is used.

        Returns
        -------
        dict
            ``{"status": "ok", "metrics": {...}, "param_count": int, "training_time_s": float}``
            or ``{"status": "error", "error": str, "traceback": str}``.
        """
        try:
            return self._run_impl(exp, epochs=epochs, name=name, data_loader=data_loader)
        except Exception as e:
            result: Dict[str, Any] = {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
            self._results.append({
                "name": name or f"experiment_{len(self._results)}",
                "config": exp,
                "result": result,
            })
            return result

    def _run_impl(
        self,
        exp: Dict[str, Any],
        epochs: int,
        name: Optional[str],
        data_loader: Any,
    ) -> Dict[str, Any]:
        """Internal implementation of run()."""
        model = self._build_model_from_experiment(exp)
        training_cfg = exp.get("training", {})
        lr = training_cfg.get("lr", 0.001)
        n_paths = training_cfg.get("n_paths", 100)
        batch_size = training_cfg.get("batch_size", 4)
        horizon = training_cfg.get("horizon", 12)
        feature_dim = exp["model"]["backbone"].get("feature_dim", 4)
        seq_len = exp["model"]["backbone"].get("seq_len", 32)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        param_count = _count_params(model)

        t_start = time.time()
        last_metrics: Dict[str, float] = {}

        for epoch in range(epochs):
            if data_loader is not None:
                # Use provided data loader
                for batch in data_loader:
                    adapted = self._adapt_batch(batch, model)
                    last_metrics = self._train_step(
                        model, optimizer, adapted, horizon, n_paths,
                    )
            else:
                # Synthetic data
                batch = _build_dummy_batch(batch_size, seq_len, feature_dim, horizon)
                last_metrics = self._train_step(
                    model, optimizer, batch, horizon, n_paths,
                )

        training_time = time.time() - t_start

        result: Dict[str, Any] = {
            "status": "ok",
            "metrics": last_metrics,
            "param_count": param_count,
            "training_time_s": round(training_time, 2),
        }

        run_name = name or f"experiment_{len(self._results)}"
        self._results.append({
            "name": run_name,
            "config": exp,
            "result": result,
        })
        return result

    def _adapt_batch(
        self, batch: Dict[str, torch.Tensor], model: SynthModel
    ) -> Dict[str, torch.Tensor]:
        """Adapt a DataLoader batch if it uses the (B, F, T) format."""
        if "history" in batch and "initial_price" in batch and "target_factors" in batch:
            return batch
        # Assume MarketDataLoader format: {"inputs": (B, F, T), "target": ...}
        adapter = DataToModelAdapter(device=next(model.parameters()).device)
        return adapter(batch)

    @staticmethod
    def _train_step(
        model: SynthModel,
        optimizer: optim.Optimizer,
        batch: Dict[str, torch.Tensor],
        horizon: int,
        n_paths: int,
    ) -> Dict[str, float]:
        """Single training step returning metric dict."""
        model.train()
        optimizer.zero_grad()

        history = batch["history"]
        initial_price = batch["initial_price"]
        target = batch["target_factors"]

        paths, mu, sigma = model(
            history, initial_price=initial_price, horizon=horizon, n_paths=n_paths,
        )
        sim_paths = prepare_paths_for_crps(paths)
        crps = crps_ensemble(sim_paths, target)
        loss = crps.mean()
        loss.backward()
        optimizer.step()

        sharpness = sim_paths.std(dim=-1).mean()
        loglik = log_likelihood(sim_paths, target).mean()
        return {
            "crps": crps.mean().item(),
            "sharpness": sharpness.item(),
            "log_likelihood": loglik.item(),
        }

    def run_preset(
        self,
        preset_name: str,
        epochs: int = 1,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a named preset with optional overrides.

        Parameters
        ----------
        preset_name : str
            Name from :meth:`list_presets` (e.g. ``"baseline_gbm"``).
        epochs : int
            Number of training epochs.
        overrides : dict, optional
            Keys to override in the default preset config (e.g.
            ``{"d_model": 64, "horizon": 24}``).

        Returns
        -------
        dict
            Same format as :meth:`run`.
        """
        preset = None
        for p in _PRESETS:
            if p["name"] == preset_name:
                preset = p
                break
        if preset is None:
            available = [p["name"] for p in _PRESETS]
            raise KeyError(f"Unknown preset '{preset_name}'. Available: {available}")

        defaults: Dict[str, Any] = {
            "d_model": 32,
            "feature_dim": 4,
            "seq_len": 32,
            "horizon": 12,
            "n_paths": 100,
            "batch_size": 4,
            "lr": 0.001,
        }
        if overrides:
            defaults.update(overrides)

        exp = self.create_experiment(
            blocks=preset["blocks"],
            head=preset["head"],
            **defaults,
        )
        return self.run(exp, epochs=epochs, name=preset_name)

    def sweep(
        self,
        preset_names: Optional[List[str]] = None,
        epochs: int = 1,
    ) -> Dict[str, Any]:
        """Run multiple presets and return a comparison.

        Parameters
        ----------
        preset_names : list[str], optional
            Preset names to sweep.  When *None*, all presets are run.
        epochs : int
            Epochs per preset.

        Returns
        -------
        dict
            Same format as :meth:`compare`, computed over just the sweep runs.
        """
        names = preset_names or [p["name"] for p in _PRESETS]
        sweep_results: List[Dict[str, Any]] = []

        for name in names:
            result = self.run_preset(name, epochs=epochs)
            sweep_results.append({
                "name": name,
                "result": result,
            })

        return self.compare()

    # ── Session state ─────────────────────────────────────────────────────

    def compare(self) -> Dict[str, Any]:
        """Rank all accumulated experiments by CRPS (best-first).

        Returns
        -------
        dict
            ``{"ranking": [{"name": str, "crps": float, "param_count": int, "experiment": dict}, ...]}``
            sorted by CRPS ascending (lower is better).
        """
        ranking: List[Dict[str, Any]] = []
        for entry in self._results:
            result = entry["result"]
            crps_val = float("inf")
            param_count = 0
            if result.get("status") == "ok":
                crps_val = result.get("metrics", {}).get("crps", float("inf"))
                param_count = result.get("param_count", 0)
            ranking.append({
                "name": entry["name"],
                "crps": crps_val,
                "param_count": param_count,
                "experiment": entry["config"],
                "status": result.get("status", "unknown"),
            })
        ranking.sort(key=lambda r: r["crps"])
        return {"ranking": ranking}

    def summary(self) -> Dict[str, Any]:
        """Return an overview of accumulated session state.

        Returns
        -------
        dict
            ``{"num_experiments": int, "results": list}``
        """
        return {
            "num_experiments": len(self._results),
            "results": [
                {
                    "name": entry["name"],
                    "status": entry["result"].get("status"),
                    "metrics": entry["result"].get("metrics"),
                    "param_count": entry["result"].get("param_count"),
                    "training_time_s": entry["result"].get("training_time_s"),
                }
                for entry in self._results
            ],
        }

    def clear(self) -> None:
        """Reset all accumulated results and free memory."""
        self._results.clear()


# ── Standalone convenience function ──────────────────────────────────────


def quick_experiment(
    blocks: List[str],
    head: str,
    d_model: int = 32,
    feature_dim: int = 4,
    seq_len: int = 32,
    horizon: int = 12,
    n_paths: int = 100,
    batch_size: int = 4,
    lr: float = 0.001,
    epochs: int = 1,
    **kwargs: Any,
) -> Dict[str, Any]:
    """One-liner to create, run, and return results for a single experiment.

    Parameters
    ----------
    blocks : list[str]
        Block names for the backbone.
    head : str
        Head registry key.
    d_model, feature_dim, seq_len, horizon, n_paths, batch_size, lr :
        Standard experiment parameters (with sensible defaults).
    epochs : int
        Number of training epochs.
    **kwargs :
        Forwarded to :meth:`ResearchSession.create_experiment` (e.g.
        ``head_kwargs``, ``block_kwargs``).

    Returns
    -------
    dict
        Run result dict (same as :meth:`ResearchSession.run`).
    """
    session = ResearchSession()
    exp = session.create_experiment(
        blocks=blocks,
        head=head,
        d_model=d_model,
        feature_dim=feature_dim,
        seq_len=seq_len,
        horizon=horizon,
        n_paths=n_paths,
        batch_size=batch_size,
        lr=lr,
        **kwargs,
    )
    return session.run(exp, epochs=epochs)


__all__ = [
    "ResearchSession",
    "quick_experiment",
]
