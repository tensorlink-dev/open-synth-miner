"""Public package surface for Open Synth Miner.

This module aggregates the most commonly used factory helpers,
registry utilities, and lightweight data loaders so downstream
projects can simply import ``open_synth_miner`` without drilling
into submodules.
"""

from src.models.factory import (
    SynthModel,
    HybridBackbone,
    ParallelFusion,
    build_model,
    create_model,
    get_model,
    simulate_gbm_paths,
)
from src.models.registry import discover_components, registry
from src.data import MarketDataLoader
from src.research.experiment_mgr import run_experiment

__all__ = [
    "SynthModel",
    "HybridBackbone",
    "ParallelFusion",
    "build_model",
    "create_model",
    "get_model",
    "simulate_gbm_paths",
    "discover_components",
    "registry",
    "MarketDataLoader",
    "run_experiment",
]
