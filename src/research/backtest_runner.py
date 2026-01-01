"""Backtesting utilities to score models across multiple horizons."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.models.factory import SynthModel
from src.research.metrics import CRPSMultiIntervalScorer, SCORING_INTERVALS, calculate_crps_for_paths
from src.tracking.wandb_logger import log_backtest_results


class BacktestRunner:
    """Run backtests for a collection of models against a dataset."""

    def __init__(
        self,
        models: Mapping[str, SynthModel],
        dataloader: DataLoader,
        scorer: CRPSMultiIntervalScorer,
        device: torch.device | str = "cpu",
    ) -> None:
        if not models:
            raise ValueError("BacktestRunner requires at least one model")
        self.models = models
        self.dataloader = dataloader
        self.scorer = scorer
        self.device = torch.device(device)

    def run(
        self,
        horizon: int,
        n_paths: int,
        *,
        log_to_wandb: bool = False,
        step: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Execute backtest and return average CRPS per interval for each model."""

        results: Dict[str, Dict[str, List[float]]] = {
            name: {interval: [] for interval in self.scorer.intervals} for name in self.models
        }

        for model in self.models.values():
            model.eval()
            model.to(self.device)

        for batch in self.dataloader:
            history = batch["history"].to(self.device)
            initial_price = batch["initial_price"].to(self.device)
            actual_series = batch.get("actual_series")
            if actual_series is None:
                raise ValueError("Backtesting requires 'actual_series' in the batch")
            actual_series = actual_series.to(self.device)

            for name, model in self.models.items():
                with torch.no_grad():
                    paths, _, _ = model(history, initial_price=initial_price, horizon=horizon, n_paths=n_paths)
                batch_results: Dict[str, List[float]] = defaultdict(list)
                for sample_idx in range(paths.shape[0]):
                    total_crps, detailed = self.scorer(paths[sample_idx], actual_series[sample_idx])
                    for row in detailed:
                        interval_name = row["Interval"]
                        if interval_name == "Overall":
                            continue
                        if row["Increment"] == "Total":
                            batch_results[interval_name].append(float(row["CRPS"]))
                for interval_name, scores in batch_results.items():
                    results[name][interval_name].extend(scores)

        averaged = {
            name: {
                interval: float(torch.tensor(scores).mean()) if scores else 0.0
                for interval, scores in intervals.items()
            }
            for name, intervals in results.items()
        }
        if log_to_wandb:
            log_backtest_results(averaged, step=step)
        return averaged
