"""Tests for edge cases in research metrics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.research.metrics import (
    calculate_crps_for_paths,
    calculate_price_changes_over_intervals,
    crps_ensemble,
    get_interval_steps,
    generate_adaptive_intervals,
    filter_valid_intervals,
    CRPSMultiIntervalScorer,
)


class TestIntervalStepsEdgeCases:
    """Test edge cases in interval step calculations."""

    def test_get_interval_steps_normal_case(self):
        """Normal case: scoring_interval is larger than time_increment."""
        # 5 minute interval, 60 second increment
        steps = get_interval_steps(scoring_interval=300, time_increment=60)
        assert steps == 5

    def test_get_interval_steps_when_increment_exceeds_interval(self):
        """Edge case: time_increment > scoring_interval."""
        # 5 minute interval, but 1 hour (3600s) increment
        steps = get_interval_steps(scoring_interval=300, time_increment=3600)
        assert steps == 0, "Should return 0 when increment exceeds interval"

    def test_calculate_price_changes_with_zero_interval_steps(self):
        """calculate_price_changes should handle interval_steps <= 0 gracefully."""
        price_paths = np.random.randn(10, 100)

        # This should not raise ValueError
        result = calculate_price_changes_over_intervals(
            price_paths, interval_steps=0, absolute_price=False
        )

        assert result.shape == (10, 0), "Should return empty array when interval_steps is 0"

    def test_calculate_price_changes_with_negative_interval_steps(self):
        """Should handle negative interval_steps (though shouldn't occur in practice)."""
        price_paths = np.random.randn(10, 100)

        result = calculate_price_changes_over_intervals(
            price_paths, interval_steps=-1, absolute_price=False
        )

        assert result.shape == (10, 0), "Should return empty array when interval_steps is negative"

    def test_crps_for_paths_handles_invalid_interval_steps(self):
        """calculate_crps_for_paths should not raise ValueError when time_increment > scoring_interval."""
        # Create sample data
        simulation_runs = np.random.randn(50, 100) + 100.0  # 50 paths, 100 timesteps
        real_price_path = np.random.randn(100) + 100.0

        # time_increment of 10000 seconds (>2.7 hours) will exceed most scoring intervals
        # This should NOT raise ValueError after fix
        total_crps, detailed = calculate_crps_for_paths(
            simulation_runs,
            real_price_path,
            time_increment=10000
        )

        assert isinstance(total_crps, float), "Should return a float"
        assert isinstance(detailed, list), "Should return a list of detailed scores"
        # Intervals with steps <= 0 should have CRPS of 0.0
        for entry in detailed:
            if entry["Increment"] == "Total":
                assert entry["CRPS"] >= 0.0, "CRPS should be non-negative or 0 for invalid intervals"


class TestCRPSEnsemble:
    """Test CRPS ensemble calculations."""

    def test_crps_ensemble_basic(self):
        """Test basic CRPS ensemble calculation."""
        # simulations: (batch, horizon, n_paths)
        simulations = torch.randn(4, 10, 50) + 100.0
        target = torch.randn(4, 10) + 100.0

        crps = crps_ensemble(simulations, target)

        assert crps.shape == (4, 10), f"Expected (4, 10), got {crps.shape}"
        assert (crps >= 0).all(), "CRPS should be non-negative"

    def test_crps_ensemble_perfect_forecast(self):
        """CRPS should be near zero when all paths match target."""
        target = torch.tensor([[100.0, 101.0, 102.0]])  # (1, 3)
        # All 50 paths are identical to target
        simulations = target.unsqueeze(-1).repeat(1, 1, 50)  # (1, 3, 50)

        crps = crps_ensemble(simulations, target)

        # CRPS should be very small (near zero) for perfect forecast
        assert (crps < 0.01).all(), f"CRPS for perfect forecast should be near 0, got {crps}"

    def test_crps_ensemble_with_single_path(self):
        """CRPS should work with n_paths=1 (deterministic forecast)."""
        simulations = torch.randn(2, 5, 1) + 100.0  # (batch=2, horizon=5, n_paths=1)
        target = torch.randn(2, 5) + 100.0

        crps = crps_ensemble(simulations, target)

        assert crps.shape == (2, 5)
        assert torch.isfinite(crps).all()


class TestPriceChangeCalculations:
    """Test price change calculation edge cases."""

    def test_price_changes_returns_mode(self):
        """Test calculation of basis point returns."""
        # Simple deterministic example
        price_paths = np.array([
            [100.0, 105.0, 110.0, 115.0],  # 5% increases
            [200.0, 210.0, 220.0, 230.0],  # 5% increases
        ])
        interval_steps = 1

        changes = calculate_price_changes_over_intervals(
            price_paths, interval_steps, absolute_price=False
        )

        # Returns in basis points: 5% = 5000 bps * (price_change/start_price)
        expected_bps = 500.0  # 5% increase = 500 bps
        assert changes.shape == (2, 3), f"Expected (2, 3), got {changes.shape}"
        assert np.allclose(changes, expected_bps, atol=1.0), f"Expected ~{expected_bps} bps, got {changes}"

    def test_price_changes_absolute_mode(self):
        """Test calculation of absolute prices."""
        price_paths = np.array([
            [100.0, 105.0, 110.0, 115.0],
            [200.0, 210.0, 220.0, 230.0],
        ])
        interval_steps = 1

        changes = calculate_price_changes_over_intervals(
            price_paths, interval_steps, absolute_price=True
        )

        # Should return prices starting from index 1
        expected = np.array([
            [105.0, 110.0, 115.0],
            [210.0, 220.0, 230.0],
        ])
        assert changes.shape == (2, 3)
        assert np.allclose(changes, expected)

    def test_price_changes_with_nan_handling(self):
        """Test that NaN values are handled correctly."""
        price_paths = np.array([
            [100.0, np.nan, 110.0, 115.0],
            [200.0, 210.0, np.nan, 230.0],
        ])
        interval_steps = 1

        # Should not raise, NaN handling is done by downstream functions
        changes = calculate_price_changes_over_intervals(
            price_paths, interval_steps, absolute_price=False
        )

        assert changes.shape == (2, 3)

    def test_price_changes_short_horizon(self):
        """Test with horizon shorter than interval_steps."""
        price_paths = np.array([
            [100.0, 105.0],  # Only 2 points
        ])
        interval_steps = 10  # Way larger than available data

        changes = calculate_price_changes_over_intervals(
            price_paths, interval_steps, absolute_price=False
        )

        # Should return empty array
        assert changes.shape == (1, 0), f"Expected (1, 0) for short horizon, got {changes.shape}"


class TestNumericalStability:
    """Test numerical stability of metrics calculations."""

    def test_crps_with_extreme_values(self):
        """CRPS should handle extreme price values without overflow."""
        # Very large prices
        simulations = torch.ones(2, 5, 50) * 1e6
        target = torch.ones(2, 5) * 1e6

        crps = crps_ensemble(simulations, target)

        assert torch.isfinite(crps).all(), "CRPS should remain finite with large values"

    def test_crps_with_zero_prices(self):
        """CRPS should handle near-zero prices."""
        simulations = torch.ones(2, 5, 50) * 1e-6
        target = torch.ones(2, 5) * 1e-6

        crps = crps_ensemble(simulations, target)

        assert torch.isfinite(crps).all(), "CRPS should remain finite with small values"


class TestAdaptiveIntervals:
    """Test adaptive interval generation and filtering."""

    def test_generate_adaptive_intervals_short_horizon(self):
        """Test generating intervals for a short 12-step horizon."""
        # 12 steps at 60 seconds each = 720 seconds (12 minutes)
        intervals = generate_adaptive_intervals(
            horizon_steps=12,
            time_increment=60,
            min_intervals=3,
        )

        assert len(intervals) >= 2, "Should generate at least 2 intervals"
        # All intervals should be <= 720 seconds
        assert all(v <= 720 for v in intervals.values()), "All intervals should fit within horizon"
        # All intervals should be > 0
        assert all(v > 0 for v in intervals.values()), "All intervals should be positive"

    def test_generate_adaptive_intervals_medium_horizon(self):
        """Test generating intervals for a 24-step horizon."""
        # 24 steps at 60 seconds each = 1440 seconds (24 minutes)
        intervals = generate_adaptive_intervals(
            horizon_steps=24,
            time_increment=60,
            min_intervals=3,
        )

        assert len(intervals) >= 3, "Should generate at least 3 intervals"
        assert all(v <= 1440 for v in intervals.values()), "All intervals should fit within horizon"

    def test_generate_adaptive_intervals_with_absolute(self):
        """Test generating intervals with absolute price option."""
        intervals = generate_adaptive_intervals(
            horizon_steps=12,
            time_increment=60,
            include_absolute=True,
        )

        # Should have at least one interval ending with "_abs"
        abs_intervals = [k for k in intervals.keys() if k.endswith("_abs")]
        assert len(abs_intervals) > 0, "Should include at least one absolute interval"

    def test_filter_valid_intervals_removes_too_large(self):
        """Test that filter_valid_intervals removes intervals larger than horizon."""
        # Horizon of 12 steps at 60s each = 720 seconds total
        base_intervals = {
            "5min": 300,      # 5 steps - VALID
            "10min": 600,     # 10 steps - VALID
            "30min": 1800,    # 30 steps - INVALID (exceeds 12 steps)
            "1hour": 3600,    # 60 steps - INVALID
        }

        valid = filter_valid_intervals(
            intervals=base_intervals,
            horizon_steps=12,
            time_increment=60,
        )

        assert "5min" in valid, "5min interval should be valid"
        assert "10min" in valid, "10min interval should be valid"
        assert "30min" not in valid, "30min interval should be filtered out"
        assert "1hour" not in valid, "1hour interval should be filtered out"

    def test_crps_scorer_adaptive_mode_short_horizon(self):
        """Test CRPSMultiIntervalScorer with adaptive=True on short horizon."""
        # Create a 12-step prediction
        torch.manual_seed(42)
        simulation_runs = torch.randn(50, 12) + 100.0  # 50 paths, 12 steps
        real_price_path = torch.randn(12) + 100.0

        # Create scorer with adaptive mode
        scorer = CRPSMultiIntervalScorer(
            time_increment=60,  # 1 minute per step
            adaptive=True,
            min_intervals=2,
        )

        total_crps, detailed = scorer(simulation_runs, real_price_path)

        # Should get non-zero scores
        assert total_crps > 0, f"Should get non-zero total CRPS, got {total_crps}"

        # Should have some interval scores
        interval_scores = [d for d in detailed if d["Increment"] == "Total" and d["Interval"] != "Overall"]
        assert len(interval_scores) >= 1, "Should have at least 1 interval score"

        # At least one interval should have non-zero CRPS
        non_zero = [d for d in interval_scores if d["CRPS"] > 0]
        assert len(non_zero) >= 1, f"Should have at least 1 non-zero interval score, got {interval_scores}"

    def test_crps_scorer_non_adaptive_mode(self):
        """Test CRPSMultiIntervalScorer with adaptive=False (original behavior)."""
        torch.manual_seed(42)
        simulation_runs = torch.randn(50, 12) + 100.0
        real_price_path = torch.randn(12) + 100.0

        # Create scorer with adaptive mode OFF
        scorer = CRPSMultiIntervalScorer(
            time_increment=60,
            adaptive=False,  # Use original fixed intervals
        )

        total_crps, detailed = scorer(simulation_runs, real_price_path)

        # This should work but may have many zero scores for intervals that don't fit
        assert isinstance(total_crps, float), "Should return a float"
        assert isinstance(detailed, list), "Should return detailed scores"

    def test_crps_scorer_caches_intervals(self):
        """Test that scorer caches adapted intervals for efficiency."""
        scorer = CRPSMultiIntervalScorer(
            time_increment=60,
            adaptive=True,
        )

        # First call for 12-step horizon
        intervals_12 = scorer.get_intervals_for_horizon(12)
        # Second call should use cache
        intervals_12_cached = scorer.get_intervals_for_horizon(12)

        assert intervals_12 is intervals_12_cached, "Should return cached intervals"

        # Different horizon should generate new intervals
        intervals_24 = scorer.get_intervals_for_horizon(24)
        assert intervals_24 is not intervals_12, "Different horizons should have different intervals"

    def test_adaptive_intervals_very_short_horizon(self):
        """Test adaptive intervals with very short horizon (edge case)."""
        intervals = generate_adaptive_intervals(
            horizon_steps=3,
            time_increment=60,
        )

        # Should still generate at least 1 interval
        assert len(intervals) >= 1, "Should generate at least 1 interval even for short horizon"

    def test_adaptive_intervals_single_step_horizon(self):
        """Test adaptive intervals with single-step horizon."""
        intervals = generate_adaptive_intervals(
            horizon_steps=1,
            time_increment=60,
        )

        # Cannot compute meaningful intervals with single step
        assert len(intervals) == 0, "Should return empty dict for single-step horizon"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
