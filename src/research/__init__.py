"""Research helpers for training and backtesting synthetic path models."""

from osa.research.ablation import AblationExperiment
from osa.research.agent_api import ResearchSession, quick_experiment
from osa.research.backtest_runner import BacktestRunner
from osa.research.experiment_mgr import load_config, run_experiment
from osa.research.metrics import CRPSMultiIntervalScorer, SCORING_INTERVALS

__all__ = [
    "AblationExperiment",
    "BacktestRunner",
    "ResearchSession",
    "run_experiment",
    "load_config",
    "quick_experiment",
    "CRPSMultiIntervalScorer",
    "SCORING_INTERVALS",
]
