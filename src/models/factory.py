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

from .backbones import BackboneBase, BlockBase
from .heads import GBMHead, HorizonHead, SimpleHorizonHead, NeuralBridgeHead, NeuralSDEHead, SDEHead, HeadBase
from .registry import discover_components

# Maximum absolute log-return for numerical stability in exp() operations
# exp(20) ≈ 4.85e8, which is safe and non-restrictive for financial returns
MAX_LOG_RETURN_CLAMP = 20.0


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
            # For concat, all dims except the last must match.
            ref = outputs[0].shape[:-1]
            for i, out in enumerate(outputs[1:], 1):
                if out.shape[:-1] != ref:
                    raise ValueError(
                        f"ParallelFusion concat: path 0 has shape {tuple(outputs[0].shape)} "
                        f"but path {i} has shape {tuple(out.shape)}. "
                        f"All dimensions except the last must match for concatenation."
                    )
            return torch.cat(outputs, dim=-1)
        # Gating: all shapes must be identical.
        ref = outputs[0].shape
        for i, out in enumerate(outputs[1:], 1):
            if out.shape != ref:
                raise ValueError(
                    f"ParallelFusion gating: path 0 has shape {tuple(outputs[0].shape)} "
                    f"but path {i} has shape {tuple(out.shape)}. "
                    f"All parallel paths must return identical shapes for gating."
                )
        weights = torch.softmax(self.gate, dim=0)
        stacked = torch.stack(outputs, dim=0)
        # Reshape weights to broadcast: (num_paths, 1, 1, ...) matching output ndim
        weight_shape = [-1] + [1] * outputs[0].ndim
        gated = (weights.view(*weight_shape) * stacked).sum(dim=0)
        return gated


class HybridBackbone(BackboneBase):
    """Backbone that stitches Hydra-instantiated blocks or callables.

    Supports automatic insertion of LayerNorm between blocks via the
    ``insert_layernorm`` parameter. When enabled, a LayerNormBlock is
    automatically inserted between each pair of consecutive blocks.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        blocks: List[Any],
        validate_shapes: bool = True,
        strict_shapes: bool = True,
        insert_layernorm: bool = False,
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
        self.insert_layernorm = insert_layernorm

        # Materialize blocks and optionally insert LayerNorm between them
        materialized_blocks = [self._materialize_block(block) for block in blocks]
        if insert_layernorm:
            materialized_blocks = self._insert_layernorm_between_blocks(materialized_blocks)

        self.layers = nn.ModuleList(materialized_blocks)
        if validate_shapes:
            self.validate_shapes(strict_shapes=strict_shapes)
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

    def _insert_layernorm_between_blocks(self, blocks: List[nn.Module]) -> List[nn.Module]:
        """Insert LayerNormBlock between consecutive blocks.

        Each LayerNormBlock normalizes activations using layer normalization
        with the model dimension.
        """
        if len(blocks) <= 1:
            return blocks

        # Import LayerNormBlock here to avoid circular imports
        from .components.advanced_blocks import LayerNormBlock

        result: List[nn.Module] = []
        for block in blocks:
            result.append(block)
            result.append(LayerNormBlock(d_model=self.d_model))

        # Remove the trailing LayerNormBlock after the last block
        result.pop()
        return result

    def _infer_output_dim(self) -> int:
        sample = torch.zeros(1, 64, self.input_size)
        with torch.no_grad():
            out = self.forward(sample)
        return out.shape[-1]

    def validate_shapes(self, strict_shapes: bool = True) -> None:
        """Ensure blocks preserve (batch, seq, d_model) shape contract.

        When blocks subclass :class:`BlockBase`, their ``min_seq_len``
        attribute is checked *before* the forward pass so that failures
        produce actionable messages instead of cryptic reshape errors.

        Blocks registered with ``preserves_seq_len=False`` (e.g.
        ``FlexiblePatchEmbed``, ``PatchEmbedding``) are allowed to change
        the sequence dimension.  The expected shape is updated accordingly
        so that downstream blocks are validated against the *actual*
        sequence length they will receive.
        """

        batch, seq = 2, 64
        expected = (batch, seq, self.d_model)
        with torch.no_grad():
            h = self.input_proj(torch.zeros(batch, seq, self.input_size))
            if h.shape != expected:
                raise ValueError(
                    "HybridBackbone input projection expected shape "
                    f"{expected} but received {tuple(h.shape)}"
                )

            for idx, layer in enumerate(self.layers):
                # Pre-check min_seq_len metadata for early, clear errors.
                min_sl = getattr(layer, "min_seq_len", 1)
                if seq < min_sl:
                    raise ValueError(
                        f"HybridBackbone block {idx} ({layer.__class__.__name__}) "
                        f"requires min_seq_len={min_sl} but the "
                        f"current sequence length is {seq}. This block is "
                        f"incompatible with very short sequences."
                    )
                h_new = layer(h)

                # Blocks marked preserves_seq_len=False may legitimately
                # change the sequence dimension (patching, pooling, etc.).
                # Update the expected shape so subsequent blocks are
                # validated against the real sequence length.
                preserves_seq = getattr(layer, "preserves_seq_len", True)
                if (
                    not preserves_seq
                    and h_new.shape[0] == batch
                    and h_new.shape[2] == self.d_model
                    and h_new.shape[1] != seq
                ):
                    seq = h_new.shape[1]
                    expected = (batch, seq, self.d_model)

                if h_new.shape != expected:
                    message = (
                        "HybridBackbone block validation: "
                        f"block {idx} ({layer.__class__.__name__}) changed shape "
                        f"from {expected} to {tuple(h_new.shape)}."
                    )
                    if strict_shapes:
                        raise ValueError(
                            message
                            + " Blocks should preserve (batch, seq, d_model) unless "
                            "dynamic shapes are explicitly enabled."
                        )
                    print(message)
                h = h_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        return h[:, -1]

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone and return the full sequence ``(batch, seq, d_model)``.

        This is used by heads that need temporal context (e.g. :class:`HorizonHead`)
        rather than the compressed last-step embedding.
        """
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        return h


class SynthModel(nn.Module):
    """Model wrapping a backbone and head for stochastic path generation.

    Expected inputs
    ----------------
    * ``x``: price/feature history shaped ``(batch, time, features)``. The
      feature dimension must align with the ``input_size`` configured on the
      backbone (see ``configs/model``), which should mirror the feature
      engineering output of the data loader.
    * ``initial_price``: scalar per batch element representing the current
      price level that anchors generated paths.

    RevIN Denormalization
    ---------------------
    If the backbone contains RevIN layers, they normalize the input features.
    During inference, the model can optionally denormalize the outputs by
    setting ``apply_revin_denorm=True``. This adjusts the predicted drift (mu)
    and volatility (sigma) to account for the input normalization, ensuring
    paths are in the correct scale.
    """

    def __init__(self, backbone: BackboneBase, head: HeadBase):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self._revin_layers = self._collect_revin_layers()

    def _collect_revin_layers(self) -> List[nn.Module]:
        """Collect all RevIN layers from the backbone for denormalization."""
        from .components.advanced_blocks import RevIN

        revin_layers = []
        if hasattr(self.backbone, "layers"):
            for layer in self.backbone.layers:
                if isinstance(layer, RevIN):
                    revin_layers.append(layer)
        return revin_layers

    def forward(
        self,
        x: torch.Tensor,
        initial_price: torch.Tensor,
        horizon: int,
        n_paths: int = 1000,
        dt: float = 1.0,
        apply_revin_denorm: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Run backbone - this may normalize via RevIN layers
        if isinstance(self.head, NeuralSDEHead):
            h_t = self.backbone(x)
            paths, mu, sigma = self.head(h_t, initial_price, horizon, n_paths, dt)
        elif isinstance(self.head, NeuralBridgeHead):
            h_t = self.backbone(x)
            macro_ret, micro_returns, sigma = self.head(h_t)
            # Apply denormalization to micro_returns and sigma if RevIN was used
            if apply_revin_denorm and self._revin_layers:
                micro_returns, sigma = self._denormalize_outputs(micro_returns, sigma)
            paths = simulate_bridge_paths(
                initial_price, micro_returns, sigma, n_paths, dt,
            )
            return paths, macro_ret.squeeze(-1), sigma
        elif isinstance(self.head, (HorizonHead, SimpleHorizonHead)):
            h_seq = self.backbone.forward_sequence(x)
            mu_seq, sigma_seq = self.head(h_seq, horizon)
            # Apply denormalization to mu_seq and sigma_seq if RevIN was used
            if apply_revin_denorm and self._revin_layers:
                mu_seq, sigma_seq = self._denormalize_outputs(mu_seq, sigma_seq)
            paths = simulate_horizon_paths(initial_price, mu_seq, sigma_seq, n_paths, dt)
            return paths, mu_seq, sigma_seq
        else:
            h_t = self.backbone(x)
            mu, sigma = self.head(h_t)
            # Apply denormalization to mu and sigma if RevIN was used
            if apply_revin_denorm and self._revin_layers:
                mu, sigma = self._denormalize_outputs(mu, sigma)
            paths = simulate_gbm_paths(initial_price, mu, sigma, horizon, n_paths, dt)
            return paths, mu, sigma

        # For NeuralSDEHead, paths are generated inside the head, so we denormalize differently
        if isinstance(self.head, NeuralSDEHead) and apply_revin_denorm and self._revin_layers:
            paths = self._denormalize_paths(paths, initial_price)
        return paths, mu, sigma

    def _denormalize_outputs(
        self, mu: torch.Tensor, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Denormalize drift and volatility using RevIN statistics.

        When RevIN normalizes the input by dividing by std, the model learns
        drift and volatility in normalized space. To get predictions in the
        original scale, we need to scale them back:
        - sigma_denorm = sigma * std (volatility scales with std)
        - mu_denorm = mu * std (drift also scales with std)

        We use the std from the first RevIN layer (if multiple exist).
        """
        if not self._revin_layers:
            return mu, sigma

        # Use the first RevIN layer's statistics
        revin = self._revin_layers[0]
        stdev = revin.stdev  # Shape: (batch, 1, d_model) or (1, 1, d_model)

        # Compute average std across features as a scalar per batch
        # This gives us a scale factor for denormalization
        scale = stdev.mean(dim=-1).squeeze(-1)  # Shape: (batch,) or scalar

        # Ensure scale is broadcastable
        if scale.ndim == 0:  # scalar
            scale = scale.item()
        elif scale.ndim == 1:  # (batch,)
            scale = scale.view(-1, 1) if mu.ndim == 2 else scale

        # Scale drift and volatility back to original space
        mu_denorm = mu * scale
        sigma_denorm = sigma * scale

        return mu_denorm, sigma_denorm

    def _denormalize_paths(
        self, paths: torch.Tensor, initial_price: torch.Tensor
    ) -> torch.Tensor:
        """Denormalize paths for NeuralSDEHead (which generates paths internally).

        For NeuralSDEHead, paths are generated inside the head using internal
        SDE integration. We denormalize by scaling the log-returns.
        """
        if not self._revin_layers:
            return paths

        # Use the first RevIN layer's statistics
        revin = self._revin_layers[0]
        stdev = revin.stdev
        scale = stdev.mean(dim=-1).squeeze(-1)

        # Convert paths to log-returns, scale, and convert back
        # paths shape: (batch, n_paths, horizon)
        log_returns = torch.log(paths / initial_price.view(-1, 1, 1))
        log_returns_scaled = log_returns * scale.view(-1, 1, 1)
        paths_denorm = initial_price.view(-1, 1, 1) * torch.exp(log_returns_scaled)

        return paths_denorm


def simulate_gbm_paths(
    initial_price: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    horizon: int,
    n_paths: int,
    dt: float = 1.0,
) -> torch.Tensor:
    """Geometric Brownian Motion path simulation with constant drift and volatility.

    Parameters
    ----------
    initial_price : (batch,)
    mu : (batch,) — constant drift
    sigma : (batch,) — constant volatility
    horizon : number of time steps
    n_paths : number of Monte Carlo paths
    dt : time-step scale

    Returns
    -------
    paths : (batch, n_paths, horizon)
    """
    batch = initial_price.shape[0]
    initial_price = initial_price.view(batch, 1, 1)
    mu = mu.view(batch, 1, 1)
    sigma = sigma.view(batch, 1, 1)

    eps = torch.randn(batch, n_paths, horizon, device=initial_price.device)
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * torch.sqrt(torch.tensor(dt, device=mu.device)) * eps
    log_returns = drift + diffusion
    cum_log_returns = torch.cumsum(log_returns, dim=2)
    cum_log_returns = torch.clamp(cum_log_returns, min=-80.0, max=80.0)
    paths = initial_price * torch.exp(cum_log_returns)
    return paths


def simulate_horizon_paths(
    initial_price: torch.Tensor,
    mu_seq: torch.Tensor,
    sigma_seq: torch.Tensor,
    n_paths: int,
    dt: float = 1.0,
) -> torch.Tensor:
    """GBM path simulation with **per-step** drift and volatility.

    Unlike :func:`simulate_gbm_paths` which uses a single ``(mu, sigma)``
    constant across all horizon steps, this variant accepts time-varying
    parameters produced by :class:`HorizonHead`.

    Parameters
    ----------
    initial_price : (batch,)
    mu_seq : (batch, horizon) — drift per step
    sigma_seq : (batch, horizon) — volatility per step
    n_paths : number of Monte-Carlo paths
    dt : time-step scale

    Returns
    -------
    paths : (batch, n_paths, horizon)
    """
    batch, horizon = mu_seq.shape
    device = mu_seq.device

    mu = mu_seq.unsqueeze(1)         # (batch, 1, horizon)
    sigma = sigma_seq.unsqueeze(1)   # (batch, 1, horizon)

    eps = torch.randn(batch, n_paths, horizon, device=device)
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * sqrt_dt * eps
    log_returns = drift + diffusion
    cum_log_returns = torch.cumsum(log_returns, dim=2)
    cum_log_returns = torch.clamp(cum_log_returns, min=-80.0, max=80.0)

    initial_price = initial_price.view(batch, 1, 1)
    paths = initial_price * torch.exp(cum_log_returns)
    return paths


def simulate_bridge_paths(
    initial_price: torch.Tensor,
    micro_returns: torch.Tensor,
    sigma: torch.Tensor,
    n_paths: int,
    dt: float = 1.0,
) -> torch.Tensor:
    """Monte-Carlo paths around a NeuralBridge mean trajectory.

    Parameters
    ----------
    initial_price : (batch,)
    micro_returns : (batch, micro_steps) — mean cumulative log-returns from the bridge head
    sigma : (batch,) — volatility scale
    n_paths : number of stochastic paths to generate
    dt : time-step scale

    Returns
    -------
    paths : (batch, n_paths, micro_steps)
    """
    batch, micro_steps = micro_returns.shape
    device = micro_returns.device

    mu = micro_returns.unsqueeze(1)          # (batch, 1, micro_steps)
    s = sigma.view(batch, 1, 1)              # (batch, 1, 1)

    eps = torch.randn(batch, n_paths, micro_steps, device=device)
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))
    log_returns = mu + s * sqrt_dt * eps
    log_returns = torch.clamp(log_returns, min=-MAX_LOG_RETURN_CLAMP, max=MAX_LOG_RETURN_CLAMP)

    initial_price = initial_price.view(batch, 1, 1)
    paths = initial_price * torch.exp(log_returns)
    return paths


HEAD_REGISTRY = {
    "gbm": GBMHead,
    "sde": SDEHead,
    "neural_sde": NeuralSDEHead,
    "horizon": HorizonHead,
    "simple_horizon": SimpleHorizonHead,
    "neural_bridge": NeuralBridgeHead,
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
    elif isinstance(head_cfg, dict) and head_cfg.get("latent_size") is not None:
        declared = head_cfg["latent_size"]
        if declared != latent_size:
            raise ValueError(
                f"Head latent_size ({declared}) does not match backbone output_dim "
                f"({latent_size}). Either remove latent_size from the head config to "
                f"auto-inject it, or set it to {latent_size}."
            )

    head = _maybe_instantiate(head_cfg_resolved)
    if not isinstance(head, HeadBase):
        head = instantiate(OmegaConf.create(head_cfg_resolved))

    return SynthModel(backbone=backbone, head=head)


def _smoke_test_model(model: SynthModel, input_size: int, seq_len: int = 64) -> None:
    """Run a tiny forward pass to surface shape errors at construction time.

    This catches feature_dim/input_size mismatches, latent_size/d_model
    inconsistencies, head routing bugs, and broken block contracts — all
    before any real data reaches the model.
    """
    batch, n_paths, horizon = 2, 2, 4
    x = torch.zeros(batch, seq_len, input_size)
    price = torch.ones(batch)
    try:
        with torch.no_grad():
            paths, mu, sigma = model(x, price, horizon=horizon, n_paths=n_paths)
        if paths.ndim != 3:
            raise ValueError(
                f"Model output should be 3D (batch, n_paths, horizon) but got "
                f"{paths.ndim}D tensor with shape {tuple(paths.shape)}."
            )
        if paths.shape[0] != batch or paths.shape[1] != n_paths:
            raise ValueError(
                f"Model output shape {tuple(paths.shape)} does not match expected "
                f"({batch}, {n_paths}, horizon)."
            )
    except Exception as e:
        raise ValueError(
            f"Model shape smoke test failed — check that your config dimensions "
            f"are consistent (input_size, d_model, latent_size, horizon_max): {e}"
        ) from e


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
        model = instantiate(model_cfg if isinstance(model_cfg, DictConfig) else OmegaConf.create(model_cfg))
    else:
        model = build_model(model_cfg)
        if isinstance(model, (DictConfig, dict)):
            raise TypeError(
                "create_model returned a configuration instead of a module. "
                "Include `_target_: src.models.factory.SynthModel` in your model config "
                "or pass an already-instantiated SynthModel."
            )

    # Smoke-test: catch shape mismatches before any real data.
    input_size = getattr(model.backbone, "input_size", None)
    if input_size is not None:
        _smoke_test_model(model, input_size)

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
    "simulate_horizon_paths",
    "simulate_bridge_paths",
    "build_model",
    "create_model",
    "get_model",
    "_smoke_test_model",
]
