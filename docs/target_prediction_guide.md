# Understanding Target Prediction in Custom Engineers

## Overview

The `FeatureEngineer` class controls **both** what features the model sees as input AND what it's trying to predict as the target. This is done through two separate methods:

- `make_input(cache, start, length)` - Creates input features from historical window
- `make_target(cache, start, length)` - Creates target values for prediction window

## Timeline

```
Price Series: [p1, p2, p3, ..., p100, p101, p102, ..., p124]
              |<--- input_len=100 --->|<-- pred_len=24 -->|

make_input():  computes features from [p1...p100]
make_target(): computes target from [p101...p124]
```

## What Each Built-in Engineer Predicts

### 1. ZScoreEngineer (3 features → 1 target)

**Input Features (3D):**
- Returns (log price changes)
- Short-window z-score (20-period)
- Long-window z-score (200-period)

**Target (1D):**
- **Future returns** over the prediction window

```python
# From line 112-114
def make_target(self, cache, start: int, length: int):
    target = cache["features"][start : start + length, 0:1]  # Column 0 = returns
    return torch.from_numpy(target).float().T
```

**You're predicting:** The log-returns for the next `pred_len` timesteps.

---

### 2. WaveletEngineer (5 features → 1 target)

**Input Features (5D):**
- Raw returns
- Wavelet approximation (low-frequency trend)
- Wavelet detail level L (high-frequency)
- Wavelet detail level L-1 (mid-frequency)
- Wavelet detail level L-2 (lower-mid-frequency)

**Target (1D):**
- **Future returns** over the prediction window

```python
# From line 165-167
def make_target(self, cache, start: int, length: int):
    target = cache["returns"][start : start + length]
    return torch.from_numpy(target[None, :]).float()
```

**You're predicting:** The raw returns for the next `pred_len` timesteps.

---

### 3. OHLCVEngineer (16 features → 1 target)

**Input Features (16D):**
- OHLCV: open, high, low, close, volume
- Volatility: realized_vol, parkinson_vol
- Moments: skew, kurtosis
- Microstructure: efficiency, vwap_dev, signed_vol_sum
- Candle patterns: up_wick, down_wick, body_size, clv

**Target (1D):**
- **Future returns** of resampled close prices

```python
# From line 1087-1089
def make_target(self, cache, start: int, length: int):
    target = cache["returns"][start : start + length]
    return torch.from_numpy(target[None, :]).float()
```

**You're predicting:** The log-returns of resampled bars for the next `pred_len` timesteps.

---

## Custom Target Examples

### Example 1: Predict Price Levels (Not Returns)

```python
class PriceLevelEngineer(FeatureEngineer):
    """Predict actual price levels instead of returns."""

    @property
    def feature_dim(self) -> int:
        return 1

    def prepare_cache(self, prices: np.ndarray):
        p = self.clean_prices(prices)
        return {"prices": p.astype(np.float32)}

    def make_input(self, cache, start: int, length: int):
        window = cache["prices"][start : start + length]
        return torch.from_numpy(window[None, :]).float()

    def make_target(self, cache, start: int, length: int):
        # Target is actual future prices, not returns!
        target = cache["prices"][start : start + length]
        return torch.from_numpy(target[None, :]).float()

    def get_volatility(self, cache, start: int, length: int) -> float:
        window = cache["prices"][start : start + length]
        return float(np.std(window))
```

**You're predicting:** Raw price values for the next `pred_len` timesteps.

---

### Example 2: Predict Volatility

```python
class VolatilityEngineer(FeatureEngineer):
    """Predict future volatility instead of returns."""

    def __init__(self, vol_window: int = 20):
        self.vol_window = vol_window

    @property
    def feature_dim(self) -> int:
        return 2  # returns + volatility

    def prepare_cache(self, prices: np.ndarray):
        p = self.clean_prices(prices)
        log_prices = np.log(p + 1e-12)
        returns = np.diff(log_prices, prepend=log_prices[0]).astype(np.float32)

        # Rolling volatility
        series = pd.Series(returns)
        volatility = series.rolling(self.vol_window).std().fillna(0.0).values.astype(np.float32)

        features = np.stack([returns, volatility], axis=1)
        return {"features": features, "volatility": volatility, "returns": returns}

    def make_input(self, cache, start: int, length: int):
        window = cache["features"][start : start + length]
        return torch.from_numpy(window).float().T  # (2, length)

    def make_target(self, cache, start: int, length: int):
        # Target is FUTURE VOLATILITY, not returns!
        target = cache["volatility"][start : start + length]
        return torch.from_numpy(target[None, :]).float()

    def get_volatility(self, cache, start: int, length: int) -> float:
        window = cache["returns"][start : start + length]
        return float(np.std(window))
```

**You're predicting:** Rolling volatility for the next `pred_len` timesteps.

---

### Example 3: Multi-Target Prediction

```python
class MultiTargetEngineer(FeatureEngineer):
    """Predict both returns AND volatility simultaneously."""

    @property
    def feature_dim(self) -> int:
        return 2  # Input: returns + historical vol

    def prepare_cache(self, prices: np.ndarray):
        p = self.clean_prices(prices)
        log_prices = np.log(p + 1e-12)
        returns = np.diff(log_prices, prepend=log_prices[0]).astype(np.float32)

        series = pd.Series(returns)
        volatility = series.rolling(20).std().fillna(0.0).values.astype(np.float32)

        features = np.stack([returns, volatility], axis=1)
        return {"features": features, "returns": returns, "volatility": volatility}

    def make_input(self, cache, start: int, length: int):
        window = cache["features"][start : start + length]
        return torch.from_numpy(window).float().T  # (2, length)

    def make_target(self, cache, start: int, length: int):
        # Target is BOTH returns and volatility!
        returns = cache["returns"][start : start + length]
        volatility = cache["volatility"][start : start + length]
        target = np.stack([returns, volatility], axis=0)
        return torch.from_numpy(target).float()  # (2, length)

    def get_volatility(self, cache, start: int, length: int) -> float:
        window = cache["returns"][start : start + length]
        return float(np.std(window))
```

**You're predicting:** Both returns AND volatility for the next `pred_len` timesteps.

---

## How to Know What You're Predicting

### Method 1: Inspect the Engineer Code

Look at the `make_target()` method:

```python
def make_target(self, cache, start: int, length: int):
    # Whatever is returned here is what you're predicting!
    return torch.from_numpy(...).float()
```

### Method 2: Check a Sample

```python
from src.data import MockDataSource, MarketDataLoader, ZScoreEngineer

source = MockDataSource(length=1000)
engineer = ZScoreEngineer()
loader = MarketDataLoader(
    data_source=source,
    engineer=engineer,
    assets=["BTC"],
    input_len=96,
    pred_len=24,
    batch_size=1
)

# Get a sample
sample = loader.dataset[0]
print(f"Input shape: {sample['inputs'].shape}")   # (feature_dim, input_len)
print(f"Target shape: {sample['target'].shape}")  # (target_dim, pred_len)
print(f"Input features:\n{sample['inputs']}")
print(f"Target values:\n{sample['target']}")
```

### Method 3: Check Documentation

Look at the engineer's docstring or the `feature_dim` property to understand what's being computed.

---

## Key Design Principles

1. **Separation of Concerns**: Input features and target are computed separately
2. **Temporal Safety**: Target window starts AFTER input window ends (no leakage)
3. **Flexibility**: You can predict anything derivable from historical data:
   - Returns (most common)
   - Price levels
   - Volatility
   - Volume
   - Directional moves
   - Multiple targets simultaneously

4. **Cache Efficiency**: `prepare_cache()` computes everything once, then `make_input()` and `make_target()` just slice windows

---

## Common Pitfall: Target Dimension Mismatch

**Problem:** Your model expects targets with shape `(batch, pred_len)` but your engineer returns `(batch, target_dim, pred_len)`.

**Solution:** Make sure your target shape matches what your loss function expects:

```python
# For single-target prediction (like returns)
def make_target(self, cache, start: int, length: int):
    target = cache["returns"][start : start + length]
    return torch.from_numpy(target[None, :]).float()  # (1, length)

# For multi-target prediction
def make_target(self, cache, start: int, length: int):
    target = cache["multi_targets"][start : start + length]
    return torch.from_numpy(target).float().T  # (n_targets, length)
```

---

## Summary

| Engineer | Input Features | Target | Predicting |
|----------|---------------|--------|------------|
| `ZScoreEngineer` | 3D (returns, z_short, z_long) | 1D (returns) | Future log-returns |
| `WaveletEngineer` | 5D (wavelet decomposition) | 1D (returns) | Future log-returns |
| `OHLCVEngineer` | 16D (OHLCV + microstructure) | 1D (returns) | Future log-returns of resampled bars |

**The engineer's `make_target()` method defines what you're predicting.**
