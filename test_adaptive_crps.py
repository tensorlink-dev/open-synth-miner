"""Simple test script to verify adaptive CRPS scorer functionality."""
import torch
import numpy as np

from src.research.metrics import (
    CRPSMultiIntervalScorer,
    generate_adaptive_intervals,
    filter_valid_intervals,
    SCORING_INTERVALS,
)


def test_adaptive_intervals_generation():
    """Test generating adaptive intervals for short horizon."""
    print("Testing adaptive interval generation...")

    # 12 steps at 60 seconds each = 720 seconds (12 minutes)
    intervals = generate_adaptive_intervals(
        horizon_steps=12,
        time_increment=60,
        min_intervals=3,
    )

    print(f"  Generated {len(intervals)} intervals for 12-step horizon:")
    for name, seconds in intervals.items():
        steps = seconds // 60
        print(f"    {name}: {seconds}s ({steps} steps)")

    assert len(intervals) >= 2, "Should generate at least 2 intervals"
    assert all(v <= 720 for v in intervals.values()), "All intervals should fit within horizon"
    print("  ✓ Adaptive interval generation passed")


def test_filter_valid_intervals():
    """Test filtering intervals that exceed horizon."""
    print("\nTesting interval filtering...")

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

    print(f"  Original intervals: {list(base_intervals.keys())}")
    print(f"  Valid intervals for 12-step horizon: {list(valid.keys())}")

    assert "5min" in valid, "5min interval should be valid"
    assert "10min" in valid, "10min interval should be valid"
    assert "30min" not in valid, "30min interval should be filtered out"
    assert "1hour" not in valid, "1hour interval should be filtered out"
    print("  ✓ Interval filtering passed")


def test_crps_scorer_adaptive_vs_non_adaptive():
    """Compare adaptive vs non-adaptive scoring on short horizon."""
    print("\nTesting CRPS scorer with 12-step horizon...")

    # Create a 12-step prediction
    torch.manual_seed(42)
    np.random.seed(42)

    simulation_runs = torch.randn(50, 12) * 5 + 100.0  # 50 paths, 12 steps
    real_price_path = torch.randn(12) * 5 + 100.0

    # Test with adaptive mode (new behavior)
    print("  Testing with adaptive=True...")
    scorer_adaptive = CRPSMultiIntervalScorer(
        time_increment=60,  # 1 minute per step
        adaptive=True,
        min_intervals=2,
    )

    total_crps_adaptive, detailed_adaptive = scorer_adaptive(simulation_runs, real_price_path)

    # Count non-zero interval scores
    interval_scores = [d for d in detailed_adaptive if d["Increment"] == "Total" and d["Interval"] != "Overall"]
    non_zero_adaptive = sum(1 for d in interval_scores if d["CRPS"] > 0)

    print(f"    Total CRPS: {total_crps_adaptive:.4f}")
    print(f"    Intervals evaluated: {len(interval_scores)}")
    print(f"    Non-zero scores: {non_zero_adaptive}")

    # Test with non-adaptive mode (original behavior)
    print("  Testing with adaptive=False...")
    scorer_non_adaptive = CRPSMultiIntervalScorer(
        time_increment=60,
        adaptive=False,
    )

    total_crps_non_adaptive, detailed_non_adaptive = scorer_non_adaptive(simulation_runs, real_price_path)

    interval_scores_non = [d for d in detailed_non_adaptive if d["Increment"] == "Total" and d["Interval"] != "Overall"]
    non_zero_non_adaptive = sum(1 for d in interval_scores_non if d["CRPS"] > 0)

    print(f"    Total CRPS: {total_crps_non_adaptive:.4f}")
    print(f"    Intervals evaluated: {len(interval_scores_non)}")
    print(f"    Non-zero scores: {non_zero_non_adaptive}")

    # Adaptive mode should have more non-zero scores
    print(f"\n  Improvement: {non_zero_adaptive} vs {non_zero_non_adaptive} non-zero scores")
    assert total_crps_adaptive > 0, "Adaptive scorer should get non-zero total CRPS"
    assert non_zero_adaptive >= 1, "Adaptive scorer should have at least 1 non-zero interval"
    print("  ✓ CRPS scorer comparison passed")


def test_different_horizons():
    """Test scorer adapts to different horizon lengths."""
    print("\nTesting different horizon lengths...")

    scorer = CRPSMultiIntervalScorer(
        time_increment=60,
        adaptive=True,
    )

    for horizon in [12, 24, 48, 96]:
        intervals = scorer.get_intervals_for_horizon(horizon)
        print(f"  Horizon {horizon} steps: {len(intervals)} intervals - {list(intervals.keys())}")
        assert len(intervals) >= 1, f"Should have intervals for horizon {horizon}"

    print("  ✓ Different horizon lengths passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Adaptive CRPS Multi-Interval Scorer")
    print("=" * 60)

    try:
        test_adaptive_intervals_generation()
        test_filter_valid_intervals()
        test_crps_scorer_adaptive_vs_non_adaptive()
        test_different_horizons()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nThe adaptive CRPS scorer now:")
        print("  1. Automatically filters out intervals larger than the horizon")
        print("  2. Generates appropriate intervals for any horizon length")
        print("  3. Avoids zero scores from horizon mismatches")
        print("\nUsage:")
        print("  # Default adaptive mode")
        print("  scorer = CRPSMultiIntervalScorer(time_increment=60, adaptive=True)")
        print("\n  # Custom intervals with adaptive filtering")
        print("  scorer = CRPSMultiIntervalScorer(")
        print("      time_increment=60,")
        print("      intervals={'5min': 300, '15min': 900},")
        print("      adaptive=True")
        print("  )")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
