"""Model factories, registries, and reusable building blocks."""

from src.models.backbones import BackboneBase
from src.models.factory import (
    HybridBackbone,
    ParallelFusion,
    SynthModel,
    build_model,
    create_model,
    get_model,
    simulate_gbm_paths,
)
from src.models.heads import GBMHead, HeadBase, SDEHead
from src.models.registry import (
    CustomAttention,
    GatedMLP,
    LSTMBlock,
    PatchMerging,
    SDEEvolutionBlock,
    TransformerBlock,
    discover_components,
    registry,
)

__all__ = [
    "BackboneBase",
    "HybridBackbone",
    "ParallelFusion",
    "SynthModel",
    "build_model",
    "create_model",
    "get_model",
    "simulate_gbm_paths",
    "GBMHead",
    "SDEHead",
    "HeadBase",
    "CustomAttention",
    "GatedMLP",
    "LSTMBlock",
    "PatchMerging",
    "SDEEvolutionBlock",
    "TransformerBlock",
    "discover_components",
    "registry",
]
