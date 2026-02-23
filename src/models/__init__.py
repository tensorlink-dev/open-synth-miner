"""Model factories, registries, and reusable building blocks.

Only lightweight helpers are eagerly imported here. Individual blocks/components
are resolved lazily from the registry via ``__getattr__`` so thousands of
registrations can coexist without upfront import costs.
"""

from osa.models.backbones import BackboneBase
from osa.models.factory import (
    HybridBackbone,
    ParallelFusion,
    SynthModel,
    build_model,
    create_model,
    get_model,
    simulate_gbm_paths,
)
from osa.models.heads import GBMHead, HeadBase, NeuralSDEHead, SDEHead
from osa.models.registry import discover_components, registry

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
    "NeuralSDEHead",
    "SDEHead",
    "HeadBase",
    "discover_components",
    "registry",
]


def __getattr__(name: str):
    """Lazily expose registered blocks/components/hybrids by attribute.

    This avoids importing every block up front while still allowing convenient
    access, e.g., ``from osa import models; models.CustomBlock`` after the
    relevant modules have been discovered/registered.
    """

    lower = name.lower()
    if lower in registry.components:
        return registry.get_component(name)
    if lower in registry.blocks:
        return registry.get_block(name)
    if lower in registry.hybrids:
        return registry.get_hybrid(name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
