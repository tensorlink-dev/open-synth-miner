"""Research helpers for training and backtesting synthetic path models."""

from osa.research.ablation import AblationExperiment
from osa.research.backtest_runner import BacktestRunner
from osa.research.experiment_mgr import load_config, run_experiment
from osa.research.metrics import CRPSMultiIntervalScorer, SCORING_INTERVALS

__all__ = [
    "AblationExperiment",
    "BacktestRunner",
    "run_experiment",
    "load_config",
    "CRPSMultiIntervalScorer",
    "SCORING_INTERVALS",
]
