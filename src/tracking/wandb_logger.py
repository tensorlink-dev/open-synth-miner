"""W&B logging utilities for Synth miner experiments."""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import wandb


def _fan_chart_table(paths: torch.Tensor, actual: torch.Tensor, horizon: int) -> wandb.Table:
    """Create a fan chart table with percentile bands.

    When inputs carry a batch dimension the first element is used so the chart
    represents a single concrete example rather than a batch aggregate.

    Parameters
    ----------
    paths : (batch, n_paths, horizon) or (n_paths, horizon)
    actual : (batch, horizon) or (horizon,)
    horizon : number of forecast steps
    """
    # Collapse optional batch dimension — fan chart is per-sequence.
    if paths.ndim == 3:
        paths = paths[0]   # (n_paths, horizon)
    if actual.ndim == 2:
        actual = actual[0]  # (horizon,)

    percentiles = torch.tensor([5.0, 50.0, 95.0], device=paths.device)
    # Quantile over n_paths (dim=0) → (3, horizon)
    percentile_values = torch.quantile(paths, percentiles / 100.0, dim=0)
    data = []
    for t in range(horizon):
        row = {
            "timestep": t + 1,
            "actual_price": actual[t].item() if actual.ndim > 0 else actual.item(),
            "p5": percentile_values[0, t].item(),
            "p50": percentile_values[1, t].item(),
            "p95": percentile_values[2, t].item(),
        }
        data.append(row)
    table = wandb.Table(columns=list(data[0].keys()))
    for row in data:
        table.add_data(*row.values())
    return table


def log_experiment_results(
    metrics: Dict[str, float],
    paths: torch.Tensor,
    actual_prices: torch.Tensor,
    horizon: int,
    step: Optional[int] = None,
) -> None:
    """Log scalar metrics, fan chart, and histogram to W&B."""
    wandb.log(metrics, step=step)

    fan_chart = _fan_chart_table(paths, actual_prices, horizon)
    wandb.log({"fan_chart": fan_chart}, step=step)

    final_prices = paths[:, :, -1].flatten().cpu().numpy()
    wandb.log({"final_price_distribution": wandb.Histogram(final_prices)}, step=step)


def log_backtest_results(
    scores: Mapping[str, Mapping[str, float]],
    step: Optional[int] = None,
) -> None:
    """Log backtest CRPS scores as scalars and a W&B table."""
    scalar_metrics: Dict[str, float] = {}
    table = wandb.Table(columns=["model", "interval", "crps"])

    for model_name, interval_scores in scores.items():
        for interval, score in sorted(interval_scores.items()):
            metric_name = f"backtest/{model_name}/{interval}"
            scalar_metrics[metric_name] = score
            table.add_data(model_name, interval, score)

    if scalar_metrics:
        wandb.log(scalar_metrics, step=step)
    wandb.log({"backtest_crps": table}, step=step)
