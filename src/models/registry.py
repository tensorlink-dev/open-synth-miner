"""Composable registry for hybrid backbones and components with auto-discovery."""
from __future__ import annotations

import hashlib
import importlib
import json
import pathlib
from typing import Any, Callable, Dict, Iterable, Optional, Type

import torch
import torch.nn as nn


class Registry:
    """Simple hierarchical registry for components, blocks, and hybrids."""

    def __init__(self) -> None:
        self.components: Dict[str, Type[nn.Module]] = {}
        self.blocks: Dict[str, Type[nn.Module]] = {}
        self.hybrids: Dict[str, Callable[..., nn.Module]] = {}

    def __getattr__(self, name: str) -> Callable[..., nn.Module]:
        """Provide attribute-style access to registered entries.

        This makes registry instances compatible with Hydra dotted-path lookups
        such as ``src.models.registry.registry.TransformerBlock`` when configs
        reference the registry object directly.
        """

        lower = name.lower()
        if lower in self.components:
            return self.get_component(name)
        if lower in self.blocks:
            return self.get_block(name)
        if lower in self.hybrids:
            return self.get_hybrid(name)
        raise AttributeError(f"'Registry' object has no attribute '{name}'")

    def _ensure_unique(self, store: Dict[str, Any], key: str) -> None:
        if key in store:
            raise ValueError(f"Duplicate registration detected for '{key}'")

    def register_component(self, name: Optional[str] = None) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
            key = (name or cls.__name__).lower()
            self._ensure_unique(self.components, key)
            self.components[key] = cls
            return cls

        return decorator

    def register_block(self, name: Optional[str] = None) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
            key = (name or cls.__name__).lower()
            self._ensure_unique(self.blocks, key)
            self.blocks[key] = cls
            return cls

        return decorator

    def register_hybrid(self, name: Optional[str] = None) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
        def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
            key = (name or fn.__name__).lower()
            self._ensure_unique(self.hybrids, key)
            self.hybrids[key] = fn
            return fn

        return decorator

    def get_component(self, name: str) -> Type[nn.Module]:
        key = name.lower()
        if key not in self.components:
            raise KeyError(f"Component '{name}' not found in registry")
        return self.components[key]

    def get_block(self, name: str) -> Type[nn.Module]:
        key = name.lower()
        if key not in self.blocks:
            raise KeyError(f"Block '{name}' not found in registry")
        return self.blocks[key]

    def get_hybrid(self, name: str) -> Callable[..., nn.Module]:
        key = name.lower()
        if key not in self.hybrids:
            raise KeyError(f"Hybrid '{name}' not found in registry")
        return self.hybrids[key]

    @staticmethod
    def recipe_hash(recipe: Any) -> str:
        """Generate a stable hash for a recipe structure."""
        recipe_bytes = json.dumps(recipe, sort_keys=True).encode("utf-8")
        return hashlib.sha1(recipe_bytes).hexdigest()[:12]


def _resolve_src_root(path: pathlib.Path, package_root: str = "src") -> pathlib.Path:
    """Find the package root (default: src) given a descendant path."""

    for ancestor in [path] + list(path.parents):
        if ancestor.name == package_root:
            return ancestor
    raise ValueError(f"Could not locate package root '{package_root}' from path: {path}")


def discover_components(package_path: str | pathlib.Path, package_root: str = "src") -> None:
    """Recursively import modules to trigger registry decorators.

    Args:
        package_path: Filesystem path to the component/blocks tree (e.g., ``src/models/components``).
        package_root: Logical root package name (defaults to ``src``).
    """

    package_dir = pathlib.Path(package_path).resolve()
    if not package_dir.exists():
        return
    if package_dir.is_file():
        raise ValueError("discover_components expects a directory path")

    src_root = _resolve_src_root(package_dir, package_root=package_root)

    for path in package_dir.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        relative_parts: Iterable[str] = path.relative_to(src_root).with_suffix("").parts
        module_name = ".".join((src_root.name, *relative_parts))
        importlib.import_module(module_name)


registry = Registry()


@registry.register_component("customattention")
class CustomAttention(nn.Module):
    """Lightweight attention component used inside blocks."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return out


@registry.register_component("gatedmlp")
class GatedMLP(nn.Module):
    """Feedforward block with gating."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.fc1(x))
        z = self.dropout(z)
        z = self.fc2(z)
        gate = torch.sigmoid(self.gate).view(1, 1, -1)
        return gate * z + (1 - gate) * x


@registry.register_component("patchmerging")
class PatchMerging(nn.Module):
    """Simple downsampling over the sequence dimension."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(pooled)


@registry.register_block("transformerblock")
class TransformerBlock(nn.Module):
    """Transformer-style block combining attention and MLP."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = CustomAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = GatedMLP(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.mlp(x))
        return x


@registry.register_block("lstmblock")
class LSTMBlock(nn.Module):
    """Sequence modeling block backed by an LSTM over the projected features."""

    def __init__(self, d_model: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


@registry.register_block("sdeevolutionblock")
class SDEEvolutionBlock(nn.Module):
    """Block that learns residual stochastic updates."""

    def __init__(self, d_model: int, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


@registry.register_hybrid("attn_sde")
def attn_sde_recipe(d_model: int, **kwargs: Any) -> nn.Module:
    """Example hybrid wiring attention then SDE evolution."""

    return nn.Sequential(
        TransformerBlock(d_model=d_model, nhead=kwargs.get("nhead", 4)),
        SDEEvolutionBlock(d_model=d_model, hidden=kwargs.get("hidden", 64)),
    )


__all__ = [
    "registry",
    "Registry",
    "discover_components",
    "CustomAttention",
    "GatedMLP",
    "PatchMerging",
    "TransformerBlock",
    "LSTMBlock",
    "SDEEvolutionBlock",
]
