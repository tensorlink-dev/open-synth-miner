# Architecture Guide

## Overview

Open Synth Miner implements a modular architecture for time series forecasting with probabilistic path generation. The codebase follows strict separation of concerns with pluggable extension points.

## Core Design Principles

### 1. Tensor Shape Contracts

**Philosophy:** Enforce shape contracts at model boundaries, not at usage sites.

Models must return consistent tensor shapes:
- **Heads** return parameters; shapes are head-specific (see below)
- **SynthModel.forward()** returns `(batch, n_paths, horizon)` for **all head types**
- **Downstream code** (Trainer, metrics) can assume 3D path tensors

**Shape validation happens at the SynthModel level, not at usage sites.**

#### Head Output Contracts

Different heads return different parameter sets:

| Head Type | Return Signature | Description |
|-----------|------------------|-------------|
| `GBMHead` | `(mu, sigma)` | Constant drift and volatility scalars |
| `SDEHead` | `(mu, sigma)` | SDE parameters via deeper network |
| `HorizonHead` | `(mu_seq, sigma_seq)` | Per-step drift and volatility sequences (with cross-attention) |
| `SimpleHorizonHead` | `(mu_seq, sigma_seq)` | Per-step drift and volatility sequences (pooling + MLP, no attention) |
| `NeuralBridgeHead` | `(macro_ret, micro_returns, sigma)` | Macro target, micro trajectory, volatility |

**Why variable returns?** Each head represents a different stochastic process. Rather than force a common structure, we document contracts and let `SynthModel.forward()` handle the conversion to uniform path outputs.

#### CRPS Format Conversion

CRPS expects ensemble members in the last dimension: `(batch, horizon, n_paths)`

Use `prepare_paths_for_crps()` utility:
```python
# Model returns: (batch, n_paths, horizon)
paths = model(x, initial_price, horizon, n_paths)

# Convert for CRPS: (batch, horizon, n_paths)
sim_paths = prepare_paths_for_crps(paths)
crps = crps_ensemble(sim_paths, target)
```

### 2. Extension Points via ABCs

The codebase uses Abstract Base Classes to define extension points following the Strategy pattern.

#### Model Components

- **`HeadBase`** - Simulation parameter heads
  - Implementations: `GBMHead`, `SDEHead`, `HorizonHead`, `SimpleHorizonHead`, `NeuralBridgeHead`
  - Map backbone latent representations to stochastic simulation parameters

- **`BackboneBase`** - Sequence encoders
  - Implementations: `HybridBackbone` (composable), time series models via Hydra
  - Expected shape: `(batch, seq, d_model)` → `(batch, latent_size)`

#### Data Pipeline

- **`FeatureEngineer`** - Feature transformation strategies
  - Implementations: `ZScoreEngineer`, `WaveletEngineer`, `MultiScaleEngineer`
  - Transform raw prices into model-ready tensors

- **`ClusteringStrategy`** - Regime detection backends
  - Implementations: `KMeansStrategy`, `GaussianMixtureStrategy`
  - Pluggable clustering for market regime identification

- **`FeatureStep`** - Feature engineering pipelines
  - Implementation: `OHLCVFeatureStep`
  - Transform aggregated bars into feature DataFrames

- **`AggregationStep`** - Bar aggregation strategies
  - Implementation: `OHLCVAggregation`
  - Proper OHLCV aggregation with intra-period statistics

- **`TargetBuilder`** - Target extraction
  - Implementations: `LogReturnTarget`, `RawReturnTarget`, `MultiColumnTarget`
  - Extract model targets from feature DataFrames

#### When to Add Extension Points

Add a new ABC when:
1. Multiple implementations of the same concept exist or are planned
2. The behavior needs to be swappable at runtime
3. Different strategies have the same input/output contract

**Guidelines:**
- Use `@abc.abstractmethod` for required methods
- Provide at least one reference implementation
- Add to `__all__` with semantic comment grouping
- Document expected input/output shapes in docstrings
- Include usage examples in module-level docstring

### 3. Numerical Stability

#### Clamping Strategy

All simulation functions clamp log-returns to prevent `exp()` overflow:

```python
from src.models.factory import MAX_LOG_RETURN_CLAMP

log_returns = drift + diffusion
log_returns = torch.clamp(log_returns, min=-MAX_LOG_RETURN_CLAMP, max=MAX_LOG_RETURN_CLAMP)
prices = initial_price * torch.exp(log_returns)
```

**Constant:** `MAX_LOG_RETURN_CLAMP = 20.0`
- `exp(20) ≈ 4.85e8` - safe and non-restrictive for financial returns
- `exp(-20) ≈ 2e-9` - prevents underflow to zero

**Why clamp?** During training, especially early on, models may predict extreme drift or volatility values. Without clamping:
- `exp(large_value)` → `Inf` → `NaN` in CRPS
- `exp(large_negative)` → 0 → division by zero

#### Volatility Floor

All heads add `1e-6` to volatility predictions:
```python
sigma = F.softplus(self.sigma_proj(h_t)).squeeze(-1) + 1e-6
```

This prevents division by zero and ensures numerical stability in Monte Carlo sampling.

#### Guard Clauses

Functions validate inputs before processing:
```python
if interval_steps <= 0:
    return np.full((N, 0), np.nan)  # Return empty, don't raise
```

**Pattern:** Graceful degradation over exceptions for boundary conditions.

### 4. Error Handling Patterns

#### Validation Strategy

- **State validation:** Check object state at method entry
  ```python
  if self._model is None:
      raise RuntimeError("Strategy must be fit() before predict()")
  ```

- **Input validation:** Validate shapes and ranges early
  ```python
  if not blocks:
      raise ValueError("HybridBackbone requires non-empty block list")
  ```

- **Boundary conditions:** Return sentinel values, don't raise
  ```python
  if T <= interval_steps:
      return np.full((N, 0), np.nan)
  ```

#### When to Raise vs. Degrade

**Raise exceptions when:**
- Programmer error (wrong API usage)
- Unrecoverable state (not fitted, missing data source)
- Invalid configuration (empty block list, negative dimensions)

**Degrade gracefully when:**
- Boundary conditions (empty intervals, short horizons)
- Numerical edge cases (near-zero volatility)
- Expected data limitations (missing observations)

## Module Organization

```
open-synth-miner/
├── src/
│   ├── data/              # Data loading and preprocessing
│   │   ├── loader.py      # Core data pipeline
│   │   ├── market_data_loader.py  # Leak-safe loader with feature engineering
│   │   └── regime_loader.py       # Regime-aware balanced sampling
│   ├── models/            # Model architectures
│   │   ├── factory.py     # Model factory, simulation functions
│   │   ├── heads.py       # Stochastic parameter heads
│   │   ├── backbones.py   # Sequence encoders
│   │   └── components/    # Reusable blocks (attention, patches, etc.)
│   ├── research/          # Training and evaluation
│   │   ├── trainer.py     # Training loop and adapters
│   │   ├── metrics.py     # CRPS and probabilistic metrics
│   │   └── backtest.py    # Walk-forward backtesting
│   └── tracking/          # Experiment tracking (W&B, HF Hub)
├── tests/                 # Comprehensive test suite
├── notebooks/             # Research notebooks
└── docs/                  # Architecture and API documentation
```

## Testing Philosophy

### Test Coverage Requirements

Every significant behavioral change should include tests:

1. **Shape contracts:** Verify output shapes match documented contracts
2. **Edge cases:** Test boundary conditions (zero, negative, extreme values)
3. **Numerical stability:** Test with extreme inputs that might cause NaN/Inf
4. **Integration:** Test component interactions, not just units

### Test Organization

- `test_*.py` files mirror the `src/` structure
- Each test class focuses on one aspect (shapes, edge cases, stability)
- Use descriptive test names: `test_neural_bridge_head_returns_3_values`
- Include docstrings explaining *why* the test exists

## Common Patterns

### Composable Pipelines

Use dataclasses + ABCs for configurable pipelines:

```python
@dataclass
class PipelineConfig:
    clustering: ClusteringStrategy = KMeansStrategy(n_clusters=3)
    feature_step: FeatureStep = OHLCVFeatureStep()
    target_builder: TargetBuilder = LogReturnTarget()

pipeline = run_pipeline(config, data_source)
```

### Hydra Integration

Models are instantiated via Hydra for flexibility:

```python
# Config
model:
  _target_: src.models.factory.SynthModel
  backbone:
    _target_: src.models.backbones.HybridBackbone
    blocks: [...]
  head:
    _target_: src.models.heads.NeuralBridgeHead
```

### Backward Compatibility

When changing interfaces:
1. Deprecate old interface with warnings
2. Support both old and new for one release
3. Remove deprecated interface in next major version

**Example:**
```python
def prepare_cache_from_asset(self, asset: AssetData) -> Any:
    """New interface that accepts full asset data.

    Override this for OHLCV-aware feature engineering.
    Falls back to prepare_cache() for backward compatibility.
    """
    return self.prepare_cache(asset.prices)
```

## Code Style

### Type Hints

- All public functions must have full type hints
- Use `from __future__ import annotations` for forward references
- Optional parameters: `param: Optional[Type] = None`

### Documentation

- NumPy-style docstrings with Parameters/Returns/Raises sections
- Module-level docstrings explain purpose and list main components
- Comments explain *why*, not *what*

### Naming Conventions

- **Classes:** `PascalCase`
- **Functions/methods:** `snake_case`
- **Constants:** `SCREAMING_SNAKE_CASE`
- **Private attributes:** `_leading_underscore`

## Performance Considerations

### Batch Processing

Always operate on batched tensors:
```python
# Good
paths = simulate_gbm_paths(initial_price, mu, sigma, horizon, n_paths)

# Bad - avoid loops
paths = [simulate_single_path(...) for _ in range(n_paths)]
```

### Memory Efficiency

- Use `@torch.no_grad()` for inference
- Clear gradients explicitly: `optimizer.zero_grad()`
- Detach targets: `target = batch["target"].detach()`

### Gradient Flow

Preserve gradient flow through utility functions:
```python
def prepare_paths_for_crps(paths: torch.Tensor) -> torch.Tensor:
    """Preserves requires_grad if input requires gradients."""
    return paths.transpose(1, 2)  # In-place would break gradients
```

## Migration Guide

### From Old to New Shape Handling

**Before (defensive code):**
```python
if paths.ndim == 2:
    sim_paths = paths.unsqueeze(-1)
else:
    sim_paths = paths.transpose(1, 2)
```

**After (trust the contract):**
```python
# SynthModel.forward() guarantees 3D paths
sim_paths = prepare_paths_for_crps(paths)
```

### From Magic Numbers to Named Constants

**Before:**
```python
log_returns = torch.clamp(log_returns, min=-20.0, max=20.0)
```

**After:**
```python
from src.models.factory import MAX_LOG_RETURN_CLAMP
log_returns = torch.clamp(log_returns, min=-MAX_LOG_RETURN_CLAMP, max=MAX_LOG_RETURN_CLAMP)
```

## Future Directions

### Planned Extension Points

1. **Loss Functions** - Pluggable loss beyond CRPS (Wasserstein, energy score)
2. **Samplers** - Beyond regime-balanced (importance sampling, adversarial)
3. **Data Sources** - Live market data, alternative data providers
4. **Ensemble Strategies** - Model ensembles, forecast combinations

### Architectural Debt

Items for future cleanup:

1. **Consolidate simulation functions** - Extract shared logic to base function
2. **Standardize head outputs** - Consider structured `HeadOutput` dataclass
3. **Type safety** - Add runtime shape validation in debug mode
4. **Performance profiling** - Identify and optimize bottlenecks

---

**Version:** 1.0
**Last Updated:** 2026-02-09
**Maintainers:** See CONTRIBUTING.md
