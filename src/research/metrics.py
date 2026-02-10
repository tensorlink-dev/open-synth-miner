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
    if interval_steps <= 0 or T <= interval_steps:
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
            if interval_steps <= 0:
                detailed.append({"Interval": interval_name, "Increment": "Total", "CRPS": 0.0})
                continue
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


def generate_adaptive_intervals(
    horizon_steps: int,
    time_increment: int,
    min_intervals: int = 3,
    include_absolute: bool = False,
) -> Dict[str, int]:
    """Generate scoring intervals that fit within the available horizon.

    Args:
        horizon_steps: Number of timesteps in the prediction horizon
        time_increment: Seconds per timestep
        min_intervals: Minimum number of intervals to generate (default 3)
        include_absolute: Whether to include an absolute price interval

    Returns:
        Dictionary mapping interval names to their duration in seconds
    """
    if horizon_steps <= 1:
        return {}

    total_seconds = horizon_steps * time_increment
    intervals = {}

    # Generate intervals at fractions of the total horizon
    # Use logarithmic spacing to cover short, medium, and long intervals
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0] if horizon_steps > 4 else [0.5, 1.0]

    generated_count = 0
    for frac in fractions:
        if generated_count >= min_intervals and frac == 1.0:
            break
        interval_seconds = int(total_seconds * frac)
        if interval_seconds < time_increment:
            continue

        # Create human-readable name
        if interval_seconds < 60:
            name = f"{interval_seconds}sec"
        elif interval_seconds < 3600:
            minutes = interval_seconds // 60
            name = f"{minutes}min"
        elif interval_seconds < 86400:
            hours = interval_seconds // 3600
            name = f"{hours}hour"
        else:
            days = interval_seconds // 86400
            name = f"{days}day"

        intervals[name] = interval_seconds
        generated_count += 1

    if include_absolute and generated_count > 0:
        # Add an absolute price interval at the longest available
        longest_name = max(intervals.keys(), key=lambda k: intervals[k])
        intervals[longest_name + "_abs"] = intervals[longest_name]

    return intervals


def filter_valid_intervals(
    intervals: Dict[str, int],
    horizon_steps: int,
    time_increment: int,
    min_steps_required: int = 2,
) -> Dict[str, int]:
    """Filter intervals to only those that fit within the available horizon.

    Args:
        intervals: Dictionary of interval names to durations in seconds
        horizon_steps: Number of timesteps in the prediction horizon
        time_increment: Seconds per timestep
        min_steps_required: Minimum steps needed for an interval to be valid

    Returns:
        Filtered dictionary with only valid intervals
    """
    valid = {}
    for name, interval_seconds in intervals.items():
        steps = get_interval_steps(interval_seconds, time_increment)
        # Need at least min_steps_required to compute meaningful changes
        if steps > 0 and steps < horizon_steps and horizon_steps > min_steps_required:
            valid[name] = interval_seconds
    return valid


class CRPSMultiIntervalScorer:
    """Callable helper to compute multi-interval CRPS on tensors.

    Supports automatic adaptation to different horizon lengths to avoid
    zero scores from intervals that exceed the prediction horizon.
    """

    def __init__(
        self,
        time_increment: int,
        intervals: Dict[str, int] | None = None,
        adaptive: bool = True,
        min_intervals: int = 2,
    ):
        """Initialize the CRPS multi-interval scorer.

        Args:
            time_increment: Seconds per timestep in the data
            intervals: Custom scoring intervals (name -> seconds). If None, uses SCORING_INTERVALS
            adaptive: If True, automatically filter/generate intervals based on horizon length
            min_intervals: Minimum number of intervals to use when adaptive=True
        """
        if time_increment <= 0:
            raise ValueError("time_increment must be positive seconds per step")
        self.time_increment = int(time_increment)
        self.base_intervals = intervals or SCORING_INTERVALS
        self.adaptive = adaptive
        self.min_intervals = min_intervals

        if not self.base_intervals:
            raise ValueError("At least one scoring interval is required")

        # Cache for adapted intervals per horizon length
        self._interval_cache: Dict[int, Dict[str, int]] = {}

    def get_intervals_for_horizon(self, horizon_steps: int) -> Dict[str, int]:
        """Get appropriate scoring intervals for a given horizon length.

        Args:
            horizon_steps: Number of timesteps in the prediction

        Returns:
            Dictionary of valid intervals for this horizon
        """
        if not self.adaptive:
            return self.base_intervals

        if horizon_steps in self._interval_cache:
            return self._interval_cache[horizon_steps]

        # First try filtering existing intervals
        valid = filter_valid_intervals(
            self.base_intervals,
            horizon_steps,
            self.time_increment,
            min_steps_required=2,
        )

        # If we don't have enough valid intervals, generate adaptive ones
        if len(valid) < self.min_intervals:
            valid = generate_adaptive_intervals(
                horizon_steps,
                self.time_increment,
                min_intervals=self.min_intervals,
                include_absolute=any(k.endswith("_abs") for k in self.base_intervals),
            )

        self._interval_cache[horizon_steps] = valid
        return valid

    def __call__(self, simulation_runs: torch.Tensor, real_price_path: torch.Tensor):
        """Compute CRPS across multiple intervals.

        Args:
            simulation_runs: Simulated price paths (n_paths, horizon_steps)
            real_price_path: Actual price path (horizon_steps,)

        Returns:
            Tuple of (total_crps, detailed_scores)
        """
        sim_np = simulation_runs.detach().cpu().numpy()
        real_np = real_price_path.detach().cpu().numpy()

        if self.adaptive:
            # Determine horizon from the data
            horizon_steps = real_np.shape[-1] if real_np.ndim > 0 else 0
            intervals = self.get_intervals_for_horizon(horizon_steps)

            # Temporarily override the global SCORING_INTERVALS for this call
            original_intervals = SCORING_INTERVALS.copy()
            SCORING_INTERVALS.clear()
            SCORING_INTERVALS.update(intervals)

            try:
                result = calculate_crps_for_paths(sim_np, real_np, self.time_increment)
            finally:
                # Restore original intervals
                SCORING_INTERVALS.clear()
                SCORING_INTERVALS.update(original_intervals)

            return result
        else:
            return calculate_crps_for_paths(sim_np, real_np, self.time_increment)


__all__ = [
    "crps_ensemble",
    "crps_torch_paths",
    "log_likelihood",
    "calculate_crps_for_paths",
    "calculate_price_changes_over_intervals",
    "label_observed_blocks",
    "get_interval_steps",
    "generate_adaptive_intervals",
    "filter_valid_intervals",
    "CRPSMultiIntervalScorer",
]
