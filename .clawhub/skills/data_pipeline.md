# Skill: Data Pipeline

## Purpose
Load market data, engineer features, and produce leak-safe windowed datasets for training and backtesting within Open Synth Miner.

## When to Use
- User wants to load data from Hugging Face, parquet files, or synthetic sources
- User needs to create a custom feature engineer
- User wants to set up walk-forward or holdout validation
- User asks about data leakage prevention

---

## Core Concepts

### Data Flow

```
DataSource.load_data(assets)
    │
    ▼
List[AssetData]  ← prices, timestamps, optional OHLCV covariates
    │
    ▼
FeatureEngineer.prepare_cache(prices)
    │
    ▼
Cache dict  ← pre-computed, causal features
    │
    ▼
MarketDataset  ← windowed (input, target) pairs with volatility buckets
    │
    ▼
MarketDataLoader  ← validation strategies: static_holdout, walk_forward, hybrid_nested
    │
    ▼
DataLoader  ← batched tensors ready for SynthModel
```

### Key Classes

| Class | Location | Role |
|-------|----------|------|
| `DataSource` | `src/data/market_data_loader.py` | Abstract: loads raw asset data |
| `MockDataSource` | `src/data/market_data_loader.py` | Synthetic GBM random walk |
| `HFParquetSource` | `src/data/market_data_loader.py` | Load parquet from HF Hub |
| `HFOHLCVSource` | `src/data/market_data_loader.py` | Load per-asset OHLCV from HF Hub |
| `FeatureEngineer` | `src/data/market_data_loader.py` | Abstract: transform prices → tensors |
| `ZScoreEngineer` | `src/data/market_data_loader.py` | Rolling z-score (3 features) |
| `WaveletEngineer` | `src/data/market_data_loader.py` | Wavelet decomposition (5 features) |
| `OHLCVEngineer` | `src/data/market_data_loader.py` | 16 micro-structure features |
| `AssetData` | `src/data/market_data_loader.py` | Dataclass: name, timestamps, prices, covariates |
| `MarketDataLoader` | `src/data/market_data_loader.py` | Orchestrates source + engineer + validation |

---

## Task: Load Synthetic Data for Testing

```python
from src.data import MockDataSource, ZScoreEngineer, MarketDataLoader

source = MockDataSource(length=5000, freq="5min", seed=42)
engineer = ZScoreEngineer(short_win=20, long_win=200)
loader = MarketDataLoader(
    data_source=source,
    engineer=engineer,
    assets=["BTC", "ETH"],
    input_len=96,
    pred_len=12,
    batch_size=32,
)

# Get a single sample
sample = loader.dataset[0]
print(sample["inputs"].shape)   # (3, 96)  — 3 features, 96 timesteps
print(sample["target"].shape)   # (1, 12)  — 1 target, 12 prediction steps
```

---

## Task: Load Real OHLCV Data from Hugging Face

```python
from src.data import HFOHLCVSource, OHLCVEngineer, MarketDataLoader
import pandas as pd

source = HFOHLCVSource(
    repo_id="tensorlink-dev/open-synth-training-data",
    filename_pattern="{asset}/data.parquet",
    repo_type="dataset",
)
engineer = OHLCVEngineer(resample_rule="5min")
loader = MarketDataLoader(
    data_source=source,
    engineer=engineer,
    assets=["BTC_USD"],
    input_len=96,
    pred_len=12,
    batch_size=64,
)

# Split into train/val/test
train_dl, val_dl, test_dl = loader.static_holdout(
    pd.Timestamp("2024-01-01", tz="UTC"),
    val_size=0.2,
)
```

### Hydra Config Alternative
```yaml
# configs/data/ohlcv_loader.yaml
data:
  _target_: src.data.market_data_loader.MarketDataLoader
  data_source:
    _target_: src.data.market_data_loader.HFOHLCVSource
    repo_id: tensorlink-dev/open-synth-training-data
    repo_type: dataset
  engineer:
    _target_: src.data.market_data_loader.OHLCVEngineer
    resample_rule: "5min"
  assets: ["BTC_USD"]
  input_len: 96
  pred_len: 12
  batch_size: 64
  feature_dim: 16
```

---

## Task: Create a Custom Feature Engineer

### Step-by-step

1. **Subclass FeatureEngineer**:

```python
from src.data.market_data_loader import FeatureEngineer
import numpy as np
import torch
from typing import Any


class MyEngineer(FeatureEngineer):
    """Custom feature engineer producing N features.

    Parameters
    ----------
    window : int
        Rolling window for feature computation.
    """

    def __init__(self, window: int = 30) -> None:
        self.window = window

    @property
    def feature_dim(self) -> int:
        return 4  # Must match actual output dimension

    def prepare_cache(self, prices: np.ndarray) -> dict:
        """Pre-compute ALL features causally from the full price series.

        CRITICAL: Only use backward-looking operations. Never use future data.
        """
        p = self.clean_prices(prices)  # inherited: handles NaN, non-positive
        log_prices = np.log(p + 1e-12)
        returns = np.diff(log_prices, prepend=log_prices[0]).astype(np.float32)

        # Example: rolling mean and std (causal — only past data)
        import pandas as pd
        series = pd.Series(returns)
        rolling_mean = series.rolling(self.window, min_periods=1).mean().values
        rolling_std = series.rolling(self.window, min_periods=1).std().fillna(0).values

        features = np.stack([
            returns,
            rolling_mean.astype(np.float32),
            rolling_std.astype(np.float32),
            (returns - rolling_mean).astype(np.float32),  # residual
        ], axis=1)  # (T, 4)

        return {"features": features, "returns": returns}

    def make_input(self, cache: Any, start: int, length: int) -> torch.Tensor:
        window = cache["features"][start : start + length]
        return torch.from_numpy(window).float().T  # (feature_dim, length)

    def make_target(self, cache: Any, start: int, length: int) -> torch.Tensor:
        target = cache["returns"][start : start + length]
        return torch.from_numpy(target[None, :]).float()  # (1, length)

    def get_volatility(self, cache: Any, start: int, length: int) -> float:
        window = cache["returns"][start : start + length]
        return float(np.std(window))
```

2. **For OHLCV-aware engineers**, override `prepare_cache_from_asset`:

```python
def prepare_cache_from_asset(self, asset: AssetData) -> Any:
    """Use full OHLCV data when available, fall back to prices."""
    if asset.covariates is None:
        return self.prepare_cache(asset.prices)
    # Access OHLCV via asset.covariates and asset.covariate_columns
    ...
```

### Validation Checklist
- [ ] `feature_dim` matches actual `make_input()` output shape
- [ ] `prepare_cache()` is purely causal — no future data leakage
- [ ] `make_input()` returns shape `(feature_dim, length)`
- [ ] `make_target()` returns shape `(1, length)` or `(channels, length)`
- [ ] `get_volatility()` returns a finite float
- [ ] `clean_prices()` is called on raw prices

---

## Task: Create a Custom Data Source

```python
from src.data.market_data_loader import DataSource, AssetData
import numpy as np
from typing import List


class MySource(DataSource):
    """Load data from a custom API or file format."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

    def load_data(self, assets: List[str]) -> List[AssetData]:
        results = []
        for asset in assets:
            # Load your data here
            timestamps = ...  # np.ndarray of datetime64 or pd.Timestamps
            prices = ...      # np.ndarray of float64

            results.append(AssetData(
                name=asset,
                timestamps=timestamps,
                prices=prices,
                covariate_columns=None,  # or ["open", "high", "low", "volume"]
                covariates=None,         # or np.ndarray of shape (T, n_covariates)
            ))
        return results
```

---

## Validation Strategies

### Static Holdout
```python
train_dl, val_dl, test_dl = loader.static_holdout(
    cutoff=pd.Timestamp("2024-01-01", tz="UTC"),
    val_size=0.2,
    shuffle_train=True,
)
```
- Temporal split: everything before cutoff is train+val, after is test
- Validation split is stratified by volatility bucket
- Temporal ordering is verified to prevent leakage

### Walk-Forward
```python
for train_dl, val_dl in loader.walk_forward(
    train_period=pd.Timedelta(days=30),
    val_period=pd.Timedelta(days=7),
    step_size=pd.Timedelta(days=7),
):
    # Train on train_dl, evaluate on val_dl
    pass
```
- Expanding or rolling window
- No future leakage: validation always follows training temporally

### Hybrid Nested
```python
wf_generator, holdout_dl = loader.hybrid_nested(
    holdout_fraction=0.1,
    train_period=pd.Timedelta(days=30),
    val_period=pd.Timedelta(days=7),
    step_size=pd.Timedelta(days=7),
)
for train_dl, val_dl in wf_generator:
    pass  # Walk-forward on the playground set
# Final evaluation on holdout_dl
```
- Last 10% of data held out completely
- Walk-forward happens on the remaining 90%

---

## Data Leakage Prevention

The `MarketDataLoader` enforces temporal ordering via `_assert_temporal_order()`:

1. **No overlap**: Training horizon timestamps must be strictly before validation start timestamps
2. **Gap verification**: Temporal gaps are detected and handled via `gap_handling` parameter:
   - `"error"` (default): Raises on gaps
   - `"ffill"`: Forward-fills missing observations
   - `"nan"`: Leaves gaps as NaN
3. **Causal features**: `FeatureEngineer.prepare_cache()` must only use backward-looking operations

### Anti-patterns to Avoid
- Using `pd.Series.rolling().mean()` without `min_periods` (leaks future via NaN filling)
- Normalizing features using the full dataset mean/std (use rolling or per-window stats)
- Allowing validation windows to overlap with training prediction horizons
