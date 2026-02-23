"""Tests for adaptive CRPS multi-interval scorer functionality.

Previously lived at the repo root; moved into tests/ so pytest collects it.
"""
import torch

from osa.research.metrics import (
    CRPSMultiIntervalScorer,
    SCORING_INTERVALS,
    filter_valid_intervals,
    generate_adaptive_intervals,
)


def test_adaptive_intervals_generation():
    """Adaptive intervals generated for a 12-step horizon all fit within it."""
    intervals = generate_adaptive_intervals(
        horizon_steps=12,
        time_increment=60,
        min_intervals=3,
    )
    assert len(intervals) >= 2, "Should generate at least 2 intervals"
    assert all(v <= 720 for v in intervals.values()), "All intervals should fit within horizon"


def test_filter_valid_intervals():
    """filter_valid_intervals removes entries whose step count exceeds the horizon."""
    base_intervals = {
        "5min": 300,
        "10min": 600,
        "30min": 1800,
        "1hour": 3600,
    }
    valid = filter_valid_intervals(
        intervals=base_intervals,
        horizon_steps=12,
        time_increment=60,
    )
    assert "5min" in valid
    assert "10min" in valid
    assert "30min" not in valid
    assert "1hour" not in valid


def test_crps_scorer_adaptive_produces_nonzero_scores():
    """Adaptive scorer returns at least one non-zero interval score on a 12-step horizon."""
    torch.manual_seed(42)
    simulation_runs = torch.randn(50, 12) * 5 + 100.0
    real_price_path = torch.randn(12) * 5 + 100.0

    scorer = CRPSMultiIntervalScorer(time_increment=60, adaptive=True, min_intervals=2)
    total_crps, detailed = scorer(simulation_runs, real_price_path)

    assert total_crps > 0
    interval_scores = [
        d for d in detailed
        if d["Increment"] == "Total" and d["Interval"] != "Overall"
    ]
    assert any(d["CRPS"] > 0 for d in interval_scores), (
        "At least one interval should have non-zero CRPS"
    )


def test_crps_scorer_adapts_to_multiple_horizons():
    """Scorer generates at least one interval for each common horizon length."""
    scorer = CRPSMultiIntervalScorer(time_increment=60, adaptive=True)
    for horizon in [12, 24, 48, 96]:
        intervals = scorer.get_intervals_for_horizon(horizon)
        assert len(intervals) >= 1, f"Should have intervals for horizon {horizon}"
