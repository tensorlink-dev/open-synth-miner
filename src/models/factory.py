"""Universal model factory with Hydra integration and HF loading.

Backbone registry blocks are expected to accept and return tensors shaped as
``(batch, sequence_length, d_model)``. Implementations should preserve both the
sequence length and feature dimension unless explicitly designed for dynamic
shapes, in which case validation can be opt-out.
"""
from __future__ import annotations

import inspect
import os
from functools import partial
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .backbones import BackboneBase
from .heads import GBMHead, SDEHead, HeadBase
from .registry import discover_components


class ParallelFusion(nn.Module):
    """Run multiple modules in parallel and fuse outputs by gating or concatenation."""

    def __init__(self, paths: Iterable[nn.Module], merge_strategy: str = "gating") -> None:
        super().__init__()
        self.paths = nn.ModuleList(list(paths))
        if len(self.paths) < 2:
            raise ValueError("ParallelFusion requires at least two paths")
        self.merge_strategy = merge_strategy.lower()
        if self.merge_strategy not in {"gating", "concat"}:
            raise ValueError("merge_strategy must be 'gating' or 'concat'")
        if self.merge_strategy == "gating":
            self.gate = nn.Parameter(torch.zeros(len(self.paths)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [path(x) for path in self.paths]
        if self.merge_strategy == "concat":
            return torch.cat(outputs, dim=-1)
        weights = torch.softmax(self.gate, dim=0)
        stacked = torch.stack(outputs, dim=0)
        gated = (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)
        return gated


class HybridBackbone(BackboneBase):
    """Backbone that stitches Hydra-instantiated blocks or callables."""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        blocks: List[Any],
        validate_shapes: bool = True,
        **_: Any,
    ):
        super().__init__()
        # Hydra can pass through extra keyword arguments (e.g., from config defaults);
        # consume them to keep instantiation forward-compatible.
        if not blocks:
            raise ValueError("HybridBackbone requires a non-empty block list")
        self.input_size = input_size
        self.input_proj = nn.Linear(input_size, d_model)
        self.d_model = d_model
        self.layers = nn.ModuleList([self._materialize_block(block) for block in blocks])
        if validate_shapes:
            self.validate_shapes()
        self.output_dim = self._infer_output_dim()

    def _materialize_block(self, block: Any) -> nn.Module:
        if isinstance(block, nn.Module):
            return block
        if isinstance(block, partial):
            return self._invoke_callable(block)
        if callable(block):
            return self._invoke_callable(block)
        raise TypeError(f"Unsupported block type: {type(block)}")

    def _invoke_callable(self, fn: Any) -> nn.Module:
        signature = inspect.signature(fn)
        kwargs: Dict[str, Any] = {}
        if "d_model" in signature.parameters:
            kwargs["d_model"] = self.d_model
        if "input_dim" in signature.parameters:
            kwargs["input_dim"] = self.d_model
        return fn(**kwargs) if kwargs else fn()

    def _infer_output_dim(self) -> int:
        sample = torch.zeros(1, 2, self.input_size)
        with torch.no_grad():
            out = self.forward(sample)
        return out.shape[-1]

    def validate_shapes(self) -> None:
        """Ensure blocks preserve (batch, seq, d_model) shape contract."""

        batch, seq = 2, 3
        expected = (batch, seq, self.d_model)
        with torch.no_grad():
            h = self.input_proj(torch.zeros(batch, seq, self.input_size))
            if h.shape != expected:
                raise ValueError(
                    "HybridBackbone input projection expected shape "
                    f"{expected} but received {tuple(h.shape)}"
                )

            for idx, layer in enumerate(self.layers):
                h = layer(h)
                if h.shape != expected:
                    raise ValueError(
                        "HybridBackbone block validation failed: "
                        f"block {idx} ({layer.__class__.__name__}) changed shape "
                        f"from {expected} to {tuple(h.shape)}. Blocks should "
                        "preserve (batch, seq, d_model) unless dynamic shapes "
                        "are explicitly enabled."
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        return h[:, -1]


class SynthModel(nn.Module):
    """Model wrapping a backbone and head for stochastic path generation."""

    def __init__(self, backbone: BackboneBase, head: HeadBase):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        initial_price: torch.Tensor,
        horizon: int,
        n_paths: int = 1000,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_t = self.backbone(x)
        mu, sigma = self.head(h_t)
        paths = simulate_gbm_paths(initial_price, mu, sigma, horizon, n_paths, dt)
        return paths, mu, sigma


def simulate_gbm_paths(
    initial_price: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    horizon: int,
    n_paths: int,
    dt: float = 1.0,
) -> torch.Tensor:
    batch = initial_price.shape[0]
    initial_price = initial_price.view(batch, 1, 1)
    mu = mu.view(batch, 1, 1)
    sigma = sigma.view(batch, 1, 1)

    eps = torch.randn(batch, n_paths, horizon, device=initial_price.device)
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * torch.sqrt(torch.tensor(dt, device=mu.device)) * eps
    log_returns = drift + diffusion
    steps = torch.exp(log_returns)
    paths = initial_price * torch.cumprod(steps, dim=2)
    return paths


HEAD_REGISTRY = {
    "gbm": GBMHead,
    "sde": SDEHead,
}


def _maybe_instantiate(cfg: Any) -> Any:
    if isinstance(cfg, DictConfig) and cfg.get("_target_"):
        return instantiate(cfg)
    if isinstance(cfg, dict) and "_target_" in cfg:
        return instantiate(OmegaConf.create(cfg))
    return cfg


def _to_container(cfg: Any) -> Any:
    """Convert OmegaConf nodes to plain containers for safe downstream access."""

    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


def _instantiate_architecture(model_cfg: Dict | DictConfig) -> nn.Module:
    discover_components("src/models/components")
    return _maybe_instantiate(model_cfg)


def build_model(model_cfg: Dict | DictConfig) -> SynthModel:
    if isinstance(model_cfg, SynthModel):
        return model_cfg

    model_cfg = _to_container(model_cfg)
    if isinstance(model_cfg, dict) and "_target_" in model_cfg:
        return instantiate(OmegaConf.create(model_cfg))

    backbone_cfg = _to_container(model_cfg.get("backbone", {}))
    head_cfg = _to_container(model_cfg.get("head", {}))

    backbone = _maybe_instantiate(backbone_cfg)
    if not isinstance(backbone, BackboneBase):
        backbone = instantiate(OmegaConf.create(backbone_cfg))

    latent_size = getattr(backbone, "output_dim", None) or getattr(backbone, "d_model", None)
    if latent_size is None:
        raise ValueError("Backbone must expose output_dim for head construction")

    head_cfg_resolved = head_cfg
    if isinstance(head_cfg, dict) and "latent_size" not in head_cfg:
        head_cfg_resolved = {**head_cfg, "latent_size": latent_size}

    head = _maybe_instantiate(head_cfg_resolved)
    if not isinstance(head, HeadBase):
        head = instantiate(OmegaConf.create(head_cfg_resolved))

    return SynthModel(backbone=backbone, head=head)


def create_model(cfg: DictConfig | Dict | SynthModel) -> SynthModel:
    """Instantiate a model from either a full config or a model-only node."""

    # Accept already-built modules to keep the API idempotent.
    if isinstance(cfg, SynthModel):
        return cfg

    model_cfg: Dict | DictConfig = cfg.get("model", cfg) if isinstance(cfg, (DictConfig, dict)) else cfg

    # If users omit the top-level _target_, assume SynthModel when backbone/head are present.
    if isinstance(model_cfg, DictConfig):
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
    if isinstance(model_cfg, dict) and "_target_" not in model_cfg and (
        "backbone" in model_cfg or "head" in model_cfg
    ):
        model_cfg = {"_target_": "src.models.factory.SynthModel", **model_cfg}

    if isinstance(model_cfg, (DictConfig, dict)) and "_target_" in model_cfg:
        return instantiate(model_cfg if isinstance(model_cfg, DictConfig) else OmegaConf.create(model_cfg))

    model = build_model(model_cfg)
    if isinstance(model, (DictConfig, dict)):
        raise TypeError(
            "create_model returned a configuration instead of a module. "
            "Include `_target_: src.models.factory.SynthModel` in your model config "
            "or pass an already-instantiated SynthModel."
        )
    return model


def get_model(cfg: DictConfig | Dict) -> nn.Module:
    """Universal factory that builds or loads a model based on config."""

    discover_components("src/models/components")
    model_cfg: Dict | DictConfig = cfg.get("model", cfg)
    hf_repo_id = model_cfg.get("hf_repo_id") if isinstance(model_cfg, dict) else None

    if hf_repo_id:
        architecture_cfg = model_cfg.get("architecture") or {k: v for k, v in model_cfg.items() if k not in {"hf_repo_id", "state_dict_path"}}
        architecture = _instantiate_architecture(architecture_cfg)
        local_dir = snapshot_download(repo_id=hf_repo_id, repo_type="model")
        state_path = model_cfg.get("state_dict_path") or os.path.join(local_dir, "model.pt")
        state = torch.load(state_path, map_location="cpu")
        architecture.load_state_dict(state, strict=False)
        return architecture

    if isinstance(model_cfg, DictConfig):
        return instantiate(model_cfg)
    if isinstance(model_cfg, dict) and "_target_" in model_cfg:
        return instantiate(OmegaConf.create(model_cfg))
    return build_model(model_cfg)


__all__ = [
    "SynthModel",
    "HybridBackbone",
    "ParallelFusion",
    "simulate_gbm_paths",
    "build_model",
    "create_model",
    "get_model",
]
