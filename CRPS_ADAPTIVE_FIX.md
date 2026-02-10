# CRPS Multi-Interval Scorer - Adaptive Mode Fix

## Problem

The CRPS multi-interval scorer was getting a lot of zero scores when using predictions with horizons shorter than the hardcoded interval definitions. For example:

- With a **12-step horizon** and **60-second time increments** (12 minutes total)
- The scorer tried to evaluate intervals like:
  - `5min` (5 steps) ✓ Valid
  - `30min` (30 steps) ✗ **Exceeds horizon!**
  - `3hour` (180 steps) ✗ **Exceeds horizon!**
  - `24hour` (1440 steps) ✗ **Exceeds horizon!**

When an interval requires more steps than available in the prediction, `calculate_price_changes_over_intervals()` returns an empty array, resulting in **zero CRPS scores** for that interval.

## Solution

Added **adaptive interval mode** to `CRPSMultiIntervalScorer` that automatically:

1. **Filters out intervals** that exceed the available horizon
2. **Generates appropriate intervals** dynamically based on horizon length
3. **Caches adapted intervals** for efficiency

## Usage

### Default (Adaptive Mode - Recommended)

```python
from src.research.metrics import CRPSMultiIntervalScorer

# Adaptive mode automatically adjusts intervals to fit the horizon
scorer = CRPSMultiIntervalScorer(
    time_increment=60,  # seconds per step
    adaptive=True,      # Default: automatically adapt intervals
    min_intervals=2,    # Ensure at least 2 intervals
)

# Works with any horizon length!
simulation_runs = torch.randn(50, 12)  # 50 paths, 12 steps
real_price_path = torch.randn(12)

total_crps, detailed = scorer(simulation_runs, real_price_path)
# Now gets non-zero scores adapted to 12-step horizon
```

### Custom Intervals with Adaptive Filtering

```python
# Define your own intervals, adaptive mode filters invalid ones
scorer = CRPSMultiIntervalScorer(
    time_increment=60,
    intervals={
        "5min": 300,
        "15min": 900,
        "30min": 1800,
        "1hour": 3600,
    },
    adaptive=True,  # Automatically filters out intervals > horizon
)
```

### Legacy Non-Adaptive Mode

```python
# Use original fixed intervals (may get zeros for short horizons)
scorer = CRPSMultiIntervalScorer(
    time_increment=60,
    adaptive=False,  # Use all intervals regardless of horizon
)
```

## New Utility Functions

### `generate_adaptive_intervals()`

Automatically generates intervals appropriate for a given horizon:

```python
from src.research.metrics import generate_adaptive_intervals

intervals = generate_adaptive_intervals(
    horizon_steps=12,
    time_increment=60,
    min_intervals=3,
    include_absolute=False,
)
# Returns: {'1min': 60, '3min': 180, '7min': 420, '12min': 720}
```

### `filter_valid_intervals()`

Filters existing intervals to only those that fit within the horizon:

```python
from src.research.metrics import filter_valid_intervals

base_intervals = {
    "5min": 300,
    "30min": 1800,
    "3hour": 10800,
}

valid = filter_valid_intervals(
    intervals=base_intervals,
    horizon_steps=12,
    time_increment=60,
)
# Returns: {'5min': 300}  # Only 5min fits in 12 steps
```

## Updated Default Behavior

The following modules now use `adaptive=True` by default:

1. **`src/research/ablation.py`** - BacktestRunner initialization
2. **`src/research/trainer.py`** - evaluate_and_log function

## Benefits

1. **No more zero scores** from horizon mismatches
2. **Works with any horizon length** (12, 24, 48, 96, etc.)
3. **Automatic adaptation** - no manual configuration needed
4. **Backwards compatible** - can still use fixed intervals with `adaptive=False`
5. **Efficient caching** - intervals computed once per horizon length

## Testing

Run the test suite to verify adaptive behavior:

```bash
# Run all adaptive interval tests
uv run pytest tests/test_metrics_edge_cases.py::TestAdaptiveIntervals -v

# Or run the standalone test script
uv run python test_adaptive_crps.py
```

## Migration Guide

If you're currently seeing zero CRPS scores:

**Before:**
```python
scorer = CRPSMultiIntervalScorer(time_increment=60)
# Gets zeros for 12-step horizon because most intervals exceed it
```

**After:**
```python
scorer = CRPSMultiIntervalScorer(time_increment=60, adaptive=True)
# Automatically adapts intervals to 12-step horizon, gets meaningful scores
```

No other code changes needed! The scorer will automatically detect the horizon length from the input data.
