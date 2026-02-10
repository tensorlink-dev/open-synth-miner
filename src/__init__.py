"""Public package surface for Open Synth Miner.

This module aggregates the most commonly used factory helpers,
registry utilities, data loaders, training primitives, and metrics
so downstream projects (including agent frameworks) can import
``open_synth_miner`` without drilling into submodules.
"""

from src.models.factory import (
    SynthModel,
    HybridBackbone,
    ParallelFusion,
    build_model,
    create_model,
    get_model,
    simulate_gbm_paths,
    HEAD_REGISTRY,
)
from src.models.registry import discover_components, registry
from src.data import MarketDataLoader
from src.research.experiment_mgr import run_experiment
from src.research.trainer import train_step, evaluate_and_log, prepare_paths_for_crps
from src.research.metrics import crps_ensemble, afcrps_ensemble, log_likelihood

__all__ = [
    # Models
    "SynthModel",
    "HybridBackbone",
    "ParallelFusion",
    "build_model",
    "create_model",
    "get_model",
    "simulate_gbm_paths",
    "HEAD_REGISTRY",
    # Registry
    "discover_components",
    "registry",
    # Data
    "MarketDataLoader",
    # Training
    "run_experiment",
    "train_step",
    "evaluate_and_log",
    "prepare_paths_for_crps",
    # Metrics
    "crps_ensemble",
    "afcrps_ensemble",
    "log_likelihood",
]
