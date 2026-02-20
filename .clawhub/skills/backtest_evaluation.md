# Skill: Backtest & Evaluation

## Purpose
Run champion-vs-challenger comparisons, compute multi-interval CRPS, and evaluate model quality for Bittensor SN50 scoring.

## When to Use
- User wants to compare a new model against an existing champion
- User needs to evaluate model performance on historical data
- User asks about CRPS scoring, fan charts, or variance spread
- User wants to run walk-forward backtesting

---

## Core Concepts

### Backtest Pipeline

```
Champion Model (HF Hub)     Challenger Model (local config)
        │                              │
        ▼                              ▼
    get_model(cfg)                 get_model(cfg)
        │                              │
        └──────────┬───────────────────┘
                   │
                   ▼
        ChallengerVsChampion.run()
            ├── Aligned price window
            ├── Generate paths from both models
            ├── Compute interval CRPS (5min, 30min, 3hr, 24hr)
            ├── Compute variance spread
            └── Log fan charts + metrics to W&B
```

### Key Classes

| Class | Location | Role |
|-------|----------|------|
| `ChallengerVsChampion` | `src/research/backtest.py` | Full backtest engine |
| `CRPSMultiIntervalScorer` | `src/research/metrics.py` | Multi-interval CRPS scorer |
| `StridedTimeSeriesDataset` | `src/data/base_dataset.py` | Strided windowing for backtest |

---

## Task: Run Backtest via CLI

```bash
python main.py mode=backtest \
    data=default_loader \
    backtest.champion_repo_id=username/SN50-Hybrid-Hub \
    backtest.time_increment=60 \
    backtest.horizon=12 \
    backtest.n_paths=1000 \
    backtest.device=cpu
```

---

## Task: Programmatic Backtest

```python
import torch
from src.models.factory import get_model
from src.models.registry import discover_components
from src.research.backtest import ChallengerVsChampion

discover_components("src/models/components")

# Price window for evaluation
prices = torch.tensor([100.0, 100.5, 101.2, ...])  # Historical prices

# Configs
challenger_cfg = {...}  # Your model config (with _target_ entries)
champion_cfg = {
    "model": {
        "hf_repo_id": "username/SN50-Hybrid-Hub",
        "architecture": challenger_cfg["model"],
    }
}

engine = ChallengerVsChampion(
    challenger_cfg=challenger_cfg,
    champion_cfg=champion_cfg,
    data_window=prices,
    time_increment=60,       # seconds per timestep
    horizon=12,              # prediction horizon
    n_paths=1000,            # Monte Carlo paths
    device="cpu",
)

results = engine.run(log_to_wandb=True)
# results = {
#     "champion": {"5min": 0.23, "30min": 0.45, ...},
#     "challenger": {"5min": 0.19, "30min": 0.38, ...},
#     "spread": {"variance_spread": 0.012, "crps_overlap": 0.05},
# }
```

---

## Task: Compute Multi-Interval CRPS

CRPS is computed at multiple time intervals to evaluate prediction quality at different horizons:

```python
from src.research.metrics import CRPSMultiIntervalScorer

scorer = CRPSMultiIntervalScorer(
    time_increment=60,      # seconds per step
    adaptive=True,          # Auto-adjust intervals for short horizons
    min_intervals=3,        # Minimum number of scoring intervals
)

# simulation_paths: (n_paths, horizon_steps) — simulated price paths
# actual_path: (horizon_steps,) — actual price path
total_crps, details = scorer(simulation_paths, actual_path)

# details is a list of dicts:
# [
#   {"Interval": "5min",  "Increment": 1,       "CRPS": 0.12},
#   {"Interval": "5min",  "Increment": "Total",  "CRPS": 0.45},
#   {"Interval": "30min", "Increment": 1,        "CRPS": 0.23},
#   ...
#   {"Interval": "Overall", "Increment": "Total", "CRPS": 1.23},
# ]
```

### Default Scoring Intervals

| Interval | Seconds | Description |
|----------|---------|-------------|
| `5min` | 300 | Short-term accuracy |
| `30min` | 1,800 | Medium-term accuracy |
| `3hour` | 10,800 | Long-term accuracy |
| `24hour_abs` | 86,400 | Absolute price level (normalized by last price) |

### Adaptive Intervals

When `adaptive=True`, the scorer automatically:
1. Filters out intervals that exceed the prediction horizon
2. Generates appropriate intervals if too few remain (logarithmic spacing at 10%, 25%, 50%, 75% of horizon)

---

## Task: Custom Walk-Forward Backtest

For more control than `ChallengerVsChampion`, use the `MarketDataLoader` walk-forward:

```python
import torch
import pandas as pd
from src.data import MockDataSource, ZScoreEngineer, MarketDataLoader
from src.models.factory import create_model
from src.models.registry import discover_components
from src.research.trainer import Trainer, DataToModelAdapter, prepare_paths_for_crps
from src.research.metrics import crps_ensemble

discover_components("src/models/components")

# Setup
source = MockDataSource(length=10000, freq="5min")
engineer = ZScoreEngineer()
loader = MarketDataLoader(
    data_source=source, engineer=engineer,
    assets=["BTC"], input_len=96, pred_len=12, batch_size=16,
)

# Walk-forward loop
for fold_idx, (train_dl, val_dl) in enumerate(loader.walk_forward(
    train_period=pd.Timedelta(days=14),
    val_period=pd.Timedelta(days=3),
    step_size=pd.Timedelta(days=3),
)):
    model = create_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, optimizer=optimizer, n_paths=256)

    # Train on this fold
    for epoch in range(5):
        for batch in train_dl:
            trainer.train_step(batch)

    # Validate
    val_metrics = trainer.validate(val_dl)
    print(f"Fold {fold_idx}: val_crps={val_metrics['val_crps']:.4f}")
```

---

## Metrics Reference

### CRPS (Lower is Better)
Measures the distance between the predicted CDF and the step function at the observed value. A CRPS of 0 means perfect prediction.

### Sharpness (Lower is Better)
Standard deviation of the ensemble. Lower sharpness means tighter prediction intervals. But too-tight intervals (overconfident) will have poor CRPS.

### Log-Likelihood (Higher is Better)
Estimated log-probability of the observation under a Gaussian fit to the ensemble.

### Variance Spread
Difference in terminal-price variance between two models. Useful for comparing calibration.

### CRPS Overlap
CRPS computed between the champion's and challenger's terminal-price distributions. Lower overlap CRPS means the models agree more.

---

## Understanding Backtest Results

A good challenger model should show:

1. **Lower CRPS** than champion across most intervals — more accurate predictions
2. **Reasonable sharpness** — not too wide (uninformative) or too narrow (overconfident)
3. **Consistent performance** across walk-forward folds — not overfitting
4. **Low variance spread** — similar calibration to champion
5. **Stable multi-interval scores** — no single interval dominates

### Red Flags

| Signal | Problem |
|--------|---------|
| CRPS near zero on training data but high on val | Overfitting |
| Very low sharpness + high CRPS | Overconfident, miscalibrated |
| One interval dominates total CRPS | Model fails at specific horizons |
| CRPS increases across walk-forward folds | Distribution shift sensitivity |
| Variance spread much larger than champion | Calibration issues |
