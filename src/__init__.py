"""Public package surface for Open Synth Miner (OSA — Open Synth Architecture).

This module aggregates the most commonly used factory helpers,
registry utilities, and lightweight data loaders so downstream
projects can simply import ``open_synth_miner`` or ``osa`` without
drilling into submodules.
"""

from osa.models.factory import (
    SynthModel,
    HybridBackbone,
    ParallelFusion,
    build_model,
    create_model,
    get_model,
    simulate_gbm_paths,
)
from osa.models.registry import discover_components, registry
from osa.data import MarketDataLoader
from osa.research.agent_api import ResearchSession, quick_experiment
from osa.research.experiment_mgr import run_experiment

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
    "ResearchSession",
    "quick_experiment",
    "run_experiment",
]
