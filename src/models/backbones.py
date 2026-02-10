"""Minimal backbone and block interfaces; composition happens via block recipes."""
from __future__ import annotations

import torch
import torch.nn as nn


class BlockBase(nn.Module):
    """Base interface for backbone blocks with shape contract metadata.

    All backbone blocks should subclass ``BlockBase`` (or at minimum preserve
    the ``(batch, seq, d_model)`` contract).  The class-level attributes
    below let ``HybridBackbone`` perform shape-compatibility checks at
    construction time *without* running a dummy forward pass.

    Attributes
    ----------
    preserves_seq_len : bool
        ``True`` (default) if the block outputs the same sequence length as
        its input.  Set to ``False`` for down-sampling blocks like
        ``PatchEmbedding``.
    preserves_d_model : bool
        ``True`` (default) if the block outputs the same feature dimension as
        its input.  Set to ``False`` if the block changes the feature dim.
    min_seq_len : int
        Minimum input sequence length this block can handle (default 1).
        ``HybridBackbone.validate_shapes`` uses this to give clear errors
        instead of cryptic reshape failures.
    """

    preserves_seq_len: bool = True
    preserves_d_model: bool = True
    min_seq_len: int = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class BackboneBase(nn.Module):
    """Base interface for stitched block backbones."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError
