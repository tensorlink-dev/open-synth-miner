"""Research metrics for probabilistic forecasts."""
from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Tuple

import numpy as np
import properscoring as ps
import torch

SCORING_INTERVALS: Dict[str, int] = {
    "5min": 300,
    "30min": 1800,
    "3hour": 10800,
    "24hour_abs": 86400,
}

EPS = 1e-12


def crps_ensemble(simulations: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Vectorized CRPS for ensemble forecasts using the empirical formula."""

    target = target.unsqueeze(-1)
    diff_term = torch.abs(simulations - target).mean(dim=-1)

    sims = simulations.unsqueeze(-1)
    pairwise = torch.abs(sims - sims.transpose(-1, -2)).mean(dim=(-1, -2))
    return diff_term - 0.5 * pairwise


def crps_torch_paths(simulations: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """High-performance CRPS over paths (batch, n_paths)."""

    return crps_ensemble(simulations, targets)


def log_likelihood(simulations: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Estimate log likelihood under a Gaussian fit to simulated paths."""

    mean = simulations.mean(dim=-1)
    var = simulations.var(dim=-1, unbiased=False) + EPS
    log_scale = 0.5 * torch.log(var * 2 * torch.tensor(np.pi))
    log_prob = -0.5 * ((targets - mean) ** 2) / var - log_scale
    return log_prob


def get_interval_steps(scoring_interval: int, time_increment: int) -> int:
    return int(scoring_interval / time_increment)


def label_observed_blocks(arr: np.ndarray) -> np.ndarray:
    not_nan = ~np.isnan(arr)
    block_start = not_nan & np.concatenate(([True], ~not_nan[:-1]))
    group_numbers = np.cumsum(block_start) - 1
    return np.where(not_nan, group_numbers, -1)


def calculate_price_changes_over_intervals(
    price_paths: np.ndarray,
    interval_steps: int,
    absolute_price: bool = False,
) -> np.ndarray:
    """Convert price paths into interval returns or absolute prices."""
    N, T = price_paths.shape
    if T <= interval_steps:
        return np.full((N, 0), np.nan)
    interval_prices = price_paths[:, ::interval_steps]
    if interval_prices.shape[1] < 2:
        return np.full((N, 0), np.nan)
    if absolute_price:
        return interval_prices[:, 1:]
    diffs = np.diff(interval_prices, axis=1)
    start_prices = interval_prices[:, :-1].copy()
    start_prices[start_prices == 0] = np.nan
    returns = (diffs / start_prices) * 10_000.0  # bps
    return returns


def calculate_crps_for_paths(
    simulation_runs: np.ndarray,
    real_price_path: np.ndarray,
    time_increment: int,
) -> Tuple[float, List[Dict[str, float]]]:
    """Compute CRPS over multiple intervals for backtesting."""

    detailed: List[Dict[str, float]] = []
    sum_all = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for interval_name, interval_seconds in SCORING_INTERVALS.items():
            interval_steps = get_interval_steps(interval_seconds, time_increment)
            absolute_price = interval_name.endswith("_abs")

            if absolute_price:
                while real_price_path[::interval_steps].shape[0] == 1 and interval_steps > 1:
                    interval_steps -= 1

            simulated_changes = calculate_price_changes_over_intervals(
                simulation_runs, interval_steps, absolute_price
            )
            real_changes = calculate_price_changes_over_intervals(
                real_price_path.reshape(1, -1), interval_steps, absolute_price
            )

            if real_changes.shape[1] == 0:
                detailed.append({"Interval": interval_name, "Increment": "Total", "CRPS": 0.0})
                continue

            data_blocks = label_observed_blocks(real_changes[0])
            if len(data_blocks) == 0 or np.all(data_blocks == -1):
                detailed.append({"Interval": interval_name, "Increment": "Total", "CRPS": 0.0})
                continue

            total_increment = 0
            crps_sum_this_interval = 0.0

            for block in np.unique(data_blocks):
                if block == -1:
                    continue
                mask = data_blocks == block
                sim_blk = simulated_changes[:, mask]
                real_blk = real_changes[:, mask]
                n_inc = sim_blk.shape[1]

                for t in range(n_inc):
                    forecasts = sim_blk[:, t]
                    obs = float(real_blk[0, t])
                    if np.isnan(obs):
                        continue

                    s = ps.crps_ensemble(obs, forecasts)

                    if absolute_price and real_price_path[-1] != 0:
                        s = s / real_price_path[-1] * 10_000.0

                    crps_sum_this_interval += s
                    detailed.append(
                        {
                            "Interval": interval_name,
                            "Increment": total_increment + 1,
                            "CRPS": s,
                        }
                    )
                    total_increment += 1

            sum_all += float(crps_sum_this_interval)
            detailed.append(
                {"Interval": interval_name, "Increment": "Total", "CRPS": crps_sum_this_interval}
            )

    detailed.append({"Interval": "Overall", "Increment": "Total", "CRPS": sum_all})
    return sum_all, detailed


class CRPSMultiIntervalScorer:
    """Callable helper to compute multi-interval CRPS on tensors."""

    def __init__(self, time_increment: int, intervals: Dict[str, int] | None = None):
        if time_increment <= 0:
            raise ValueError("time_increment must be positive seconds per step")
        self.time_increment = int(time_increment)
        self.intervals = intervals or SCORING_INTERVALS
        if not self.intervals:
            raise ValueError("At least one scoring interval is required")

    def __call__(self, simulation_runs: torch.Tensor, real_price_path: torch.Tensor):
        sim_np = simulation_runs.detach().cpu().numpy()
        real_np = real_price_path.detach().cpu().numpy()
        return calculate_crps_for_paths(sim_np, real_np, self.time_increment)


__all__ = [
    "crps_ensemble",
    "crps_torch_paths",
    "log_likelihood",
    "calculate_crps_for_paths",
    "calculate_price_changes_over_intervals",
    "label_observed_blocks",
    "get_interval_steps",
    "CRPSMultiIntervalScorer",
]
