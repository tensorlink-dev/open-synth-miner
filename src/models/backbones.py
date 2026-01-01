"""Minimal backbone interface; composition happens via block recipes."""
from __future__ import annotations

import torch
import torch.nn as nn


class BackboneBase(nn.Module):
    """Base interface for stitched block backbones."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError
