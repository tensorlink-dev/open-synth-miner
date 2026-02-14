"""Composable registry for hybrid backbones and components with auto-discovery."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Type

import torch
import torch.nn as nn


@dataclasses.dataclass(frozen=True)
class BlockInfo:
    """Metadata for a registered block, component, or hybrid."""

    name: str
    cls: Any  # Type[nn.Module] for blocks/components, Callable for hybrids
    kind: str  # "block", "component", or "hybrid"
    description: str = ""
    preserves_seq_len: bool = True
    preserves_d_model: bool = True
    min_seq_len: int = 1
    source_module: str = ""


class Registry:
    """Simple hierarchical registry for components, blocks, and hybrids."""

    def __init__(self) -> None:
        self.components: Dict[str, Type[nn.Module]] = {}
        self.blocks: Dict[str, Type[nn.Module]] = {}
        self.hybrids: Dict[str, Callable[..., nn.Module]] = {}
        self._info: Dict[str, BlockInfo] = {}

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

    def register_component(
        self,
        name: Optional[str] = None,
        *,
        description: str = "",
    ) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
            key = (name or cls.__name__).lower()
            self._ensure_unique(self.components, key)
            self.components[key] = cls
            desc = description or (cls.__doc__ or "").split("\n")[0].strip()
            self._info[key] = BlockInfo(
                name=key,
                cls=cls,
                kind="component",
                description=desc,
                source_module=cls.__module__,
            )
            return cls

        return decorator

    def register_block(
        self,
        name: Optional[str] = None,
        *,
        preserves_seq_len: bool = True,
        preserves_d_model: bool = True,
        min_seq_len: int = 1,
        description: str = "",
    ) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        """Register a backbone block with optional shape metadata.

        Parameters
        ----------
        name : str, optional
            Registry key (defaults to ``cls.__name__``).
        preserves_seq_len : bool
            Whether the block preserves sequence length (default True).
        preserves_d_model : bool
            Whether the block preserves the feature dimension (default True).
        min_seq_len : int
            Minimum sequence length the block can handle (default 1).
        description : str, optional
            Human-readable description of the block.  Falls back to the
            first line of the class docstring when omitted.

        The metadata is stored as class-level attributes so that
        ``HybridBackbone.validate_shapes`` can inspect blocks without
        running a forward pass.
        """
        def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
            key = (name or cls.__name__).lower()
            self._ensure_unique(self.blocks, key)
            # Stamp shape metadata onto the class for introspection.
            cls.preserves_seq_len = preserves_seq_len
            cls.preserves_d_model = preserves_d_model
            cls.min_seq_len = min_seq_len
            self.blocks[key] = cls
            desc = description or (cls.__doc__ or "").split("\n")[0].strip()
            self._info[key] = BlockInfo(
                name=key,
                cls=cls,
                kind="block",
                description=desc,
                preserves_seq_len=preserves_seq_len,
                preserves_d_model=preserves_d_model,
                min_seq_len=min_seq_len,
                source_module=cls.__module__,
            )
            return cls

        return decorator

    def register_hybrid(
        self,
        name: Optional[str] = None,
        *,
        description: str = "",
    ) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
        def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
            key = (name or fn.__name__).lower()
            self._ensure_unique(self.hybrids, key)
            self.hybrids[key] = fn
            desc = description or (fn.__doc__ or "").split("\n")[0].strip()
            self._info[key] = BlockInfo(
                name=key,
                cls=fn,
                kind="hybrid",
                description=desc,
                source_module=fn.__module__ if hasattr(fn, "__module__") else "",
            )
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

    # ------------------------------------------------------------------
    # Block registry introspection
    # ------------------------------------------------------------------

    def get_info(self, name: str) -> BlockInfo:
        """Return :class:`BlockInfo` for a registered name."""
        key = name.lower()
        if key not in self._info:
            raise KeyError(f"'{name}' not found in registry")
        return self._info[key]

    def list_blocks(self, kind: Optional[str] = None) -> List[BlockInfo]:
        """Return a list of all registered :class:`BlockInfo` entries.

        Parameters
        ----------
        kind : str, optional
            Filter by kind (``"block"``, ``"component"``, or ``"hybrid"``).
            Returns all entries when *None*.
        """
        entries = list(self._info.values())
        if kind is not None:
            entries = [e for e in entries if e.kind == kind]
        return sorted(entries, key=lambda e: (e.kind, e.name))

    def summary(self, kind: Optional[str] = None) -> str:
        """Return a human-readable summary table of registered entries.

        Parameters
        ----------
        kind : str, optional
            Filter by kind (``"block"``, ``"component"``, or ``"hybrid"``).
            Returns all entries when *None*.
        """
        entries = self.list_blocks(kind=kind)
        if not entries:
            return "No registered entries."

        # Column widths
        name_w = max(len(e.name) for e in entries)
        kind_w = max(len(e.kind) for e in entries)
        name_w = max(name_w, 4)  # minimum header width
        kind_w = max(kind_w, 4)

        lines: List[str] = []
        header = (
            f"{'Name':<{name_w}}  {'Kind':<{kind_w}}  "
            f"{'Seq':>3}  {'Dim':>3}  {'MinS':>4}  Description"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for e in entries:
            seq = "Y" if e.preserves_seq_len else "N"
            dim = "Y" if e.preserves_d_model else "N"
            lines.append(
                f"{e.name:<{name_w}}  {e.kind:<{kind_w}}  "
                f"{seq:>3}  {dim:>3}  {e.min_seq_len:>4}  {e.description}"
            )
        return "\n".join(lines)

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


@registry.register_component("customattention", description="Multi-head self-attention component")
class CustomAttention(nn.Module):
    """Lightweight attention component used inside blocks."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return out


@registry.register_component("gatedmlp", description="Gated feedforward MLP component")
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


@registry.register_component("patchmerging", description="Sequence downsampling via average pooling")
class PatchMerging(nn.Module):
    """Simple downsampling over the sequence dimension."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(pooled)


@registry.register_block("transformerblock", description="Transformer encoder with self-attention and gated MLP")
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


@registry.register_block("lstmblock", description="LSTM-based sequence modeling block")
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


@registry.register_block("sdeevolutionblock", description="Residual stochastic differential evolution block")
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


@registry.register_hybrid("attn_sde", description="Transformer attention followed by SDE evolution")
def attn_sde_recipe(d_model: int, **kwargs: Any) -> nn.Module:
    """Example hybrid wiring attention then SDE evolution."""

    return nn.Sequential(
        TransformerBlock(d_model=d_model, nhead=kwargs.get("nhead", 4)),
        SDEEvolutionBlock(d_model=d_model, hidden=kwargs.get("hidden", 64)),
    )


__all__ = [
    "registry",
    "Registry",
    "BlockInfo",
    "discover_components",
    "CustomAttention",
    "GatedMLP",
    "PatchMerging",
    "TransformerBlock",
    "LSTMBlock",
    "SDEEvolutionBlock",
]
