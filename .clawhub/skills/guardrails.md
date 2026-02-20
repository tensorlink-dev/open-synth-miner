# Skill: Guardrails & Validation

## Purpose
Validation rules, safety checks, and anti-patterns that agents must follow when modifying the Open Synth Miner codebase. This skill acts as a pre-commit checklist for all code changes.

## When to Use
- **Always** — consult before committing any code change
- After creating new blocks, heads, data sources, or feature engineers
- Before submitting a training run
- When debugging numerical instability or shape errors

---

## Mandatory Validation Steps

### 1. Shape Contract Verification

Every modification must preserve the fundamental shape contracts:

```
Backbone blocks:  (batch, seq_len, d_model) → (batch, seq_len, d_model)
                  Unless preserves_seq_len=False is declared

Backbone output:  (batch, d_model)  [last timestep extraction]
                  or (batch, seq_len, d_model) [via forward_sequence()]

Head output:      Head-specific (documented per subclass)

SynthModel output: (paths, mu, sigma)
                   paths.shape == (batch, n_paths, horizon)  ALWAYS

CRPS input:       (batch, horizon, n_paths)  [transposed from model output]

Feature engineer:  make_input → (feature_dim, seq_len)
                   make_target → (channels, pred_len)
```

**Validation command**:
```bash
python -m pytest tests/ -v -k "shape"
```

**Quick smoke test**:
```python
from src.models.factory import create_model, _smoke_test_model
model = create_model(cfg)
# _smoke_test_model runs automatically inside create_model
```

### 2. Numerical Stability Checks

| Rule | Why | Where |
|------|-----|-------|
| `sigma = F.softplus(...) + 1e-6` | Prevent division by zero in sampling | All heads |
| `torch.clamp(log_returns, -20.0, 20.0)` | Prevent `exp()` overflow/underflow | All simulation functions |
| `MAX_LOG_RETURN_CLAMP = 20.0` | `exp(20) ≈ 4.85e8`, safe limit | `src/models/factory.py` |
| `np.clip(prices, 1e-6, None)` | Prevent `log(0)` in feature engineering | `FeatureEngineer.clean_prices()` |
| `+ 1e-8` or `+ 1e-12` denominators | Prevent division by zero | Rolling stats, normalizations |

**Test for NaN/Inf**:
```python
def test_no_nan_output():
    model = create_model(cfg)
    x = torch.randn(2, 64, feature_dim)
    price = torch.ones(2)
    paths, mu, sigma = model(x, price, horizon=12, n_paths=100)
    assert not torch.isnan(paths).any(), "NaN in paths"
    assert not torch.isinf(paths).any(), "Inf in paths"
    assert (sigma > 0).all(), "Non-positive sigma"
```

### 3. Data Leakage Prevention

**Forbidden patterns in feature engineering**:

```python
# BAD: Future data leakage via global normalization
mean = prices.mean()  # Uses entire series including future
features = (prices - mean) / prices.std()

# GOOD: Causal rolling normalization
rolling_mean = pd.Series(prices).rolling(window, min_periods=1).mean()
features = (prices - rolling_mean) / (rolling_std + 1e-8)

# BAD: Centered rolling window
features = pd.Series(prices).rolling(window, center=True).mean()

# GOOD: Backward-looking only
features = pd.Series(prices).rolling(window, center=False).mean()

# BAD: Using .shift(-n) which looks into the future
features = pd.Series(prices).shift(-1)

# GOOD: Using .shift(n) which looks into the past
features = pd.Series(prices).shift(1)
```

**Temporal ordering verification**:
```python
# MarketDataLoader enforces this automatically
loader._assert_temporal_order(train_ds, val_ds)
# Raises ValueError if validation data temporally overlaps with training
```

### 4. Gradient Flow Verification

**Required for all new blocks and heads**:

```python
def test_gradient_flow():
    model = create_model(cfg)
    x = torch.randn(2, 64, feature_dim, requires_grad=True)
    price = torch.ones(2)
    paths, mu, sigma = model(x, price, horizon=12, n_paths=10)
    loss = paths.mean()
    loss.backward()
    # Verify gradients exist and are finite
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
```

**Anti-patterns that break gradients**:
```python
# BAD: Detaching inside forward pass
x = x.detach()  # Breaks gradient flow

# BAD: Using .item() in computation graph
value = x.mean().item()  # Converts to Python float, no gradient

# BAD: In-place operations
x += residual  # Use x = x + residual instead

# BAD: Non-differentiable operations in the path
x = (x > 0).float()  # Step function has zero gradient
```

---

## Pre-Change Checklist

Before modifying any source file, verify:

### For New Blocks (`src/models/components/`)

- [ ] File is in `src/models/components/` (auto-discovery requires this)
- [ ] Decorated with `@registry.register_block("name")`
- [ ] Accepts `d_model` as constructor parameter
- [ ] `forward()` input/output shapes are documented
- [ ] Shape metadata (`preserves_seq_len`, `preserves_d_model`, `min_seq_len`) is accurate
- [ ] No in-place operations on tensors
- [ ] Test file created in `tests/`
- [ ] Shape test, gradient test, and edge-case test included

### For New Heads (`src/models/heads.py`)

- [ ] Subclasses `HeadBase`
- [ ] `forward()` return type documented
- [ ] `latent_size` parameter for backbone compatibility
- [ ] Volatility floor: `+ 1e-6` on all sigma outputs
- [ ] Added to `HEAD_REGISTRY` in `factory.py`
- [ ] Routing logic added in `SynthModel.forward()` (if non-standard output)
- [ ] Corresponding simulation function exists (or uses existing one)
- [ ] Test verifying output shapes and gradient flow

### For New Feature Engineers

- [ ] Subclasses `FeatureEngineer`
- [ ] `feature_dim` property returns correct count
- [ ] `prepare_cache()` uses only backward-looking operations
- [ ] `make_input()` returns `(feature_dim, length)`
- [ ] `make_target()` returns `(channels, length)`
- [ ] `get_volatility()` returns finite float
- [ ] `clean_prices()` called on raw input
- [ ] No future data leakage (verified by temporal ordering test)
- [ ] Works with `MockDataSource` as fallback

### For New Data Sources

- [ ] Subclasses `DataSource`
- [ ] `load_data()` returns `List[AssetData]`
- [ ] Timestamps are timezone-aware (`tz="UTC"`)
- [ ] Prices are positive floats
- [ ] Handles missing assets gracefully (skip, don't crash)

### For Config Changes

- [ ] `_target_` paths are fully qualified (e.g., `src.models.factory.SynthModel`)
- [ ] `d_model` consistent across all blocks in the backbone
- [ ] `latent_size` matches backbone `d_model`
- [ ] `feature_dim` matches between data config and model config
- [ ] Config renders correctly: `python main.py --cfg job`

---

## Common Anti-Patterns

### Architecture Anti-Patterns

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Blocks with different `d_model` | Dimension mismatch crash | Use same `d_model` everywhere |
| Missing `validate_shapes: true` | Shape bugs discovered late | Always validate (default is true) |
| `ParallelFusion` with mismatched shapes | Runtime crash in gating | Ensure all paths output same shape |
| Head without `+ 1e-6` on sigma | NaN in sampling | Always add volatility floor |
| Unbounded `exp()` in forward pass | Inf/NaN overflow | Clamp log-returns first |

### Data Anti-Patterns

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Global mean/std normalization | Future data leakage | Rolling or per-window stats |
| Centered rolling windows | Future data leakage | Use `center=False` |
| Missing `min_periods=1` | NaN at series start | Always set `min_periods` |
| Not calling `clean_prices()` | NaN/negative prices | Call in `prepare_cache()` |
| Overlapping train/val windows | Data leakage | Use `MarketDataLoader` validation |

### Training Anti-Patterns

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Forgetting `model.train()` | Dropout/BN disabled | Call before training steps |
| Forgetting `model.eval()` | Dropout active during eval | Call before evaluation |
| Not using `torch.no_grad()` for eval | Memory waste | Wrap eval in `torch.no_grad()` |
| `optimizer.step()` before `loss.backward()` | No gradient update | Always backward then step |
| Very large `n_paths` in training | OOM | Start with 256, increase if needed |

---

## Test Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_hybrid_backbone.py -v

# Run tests matching a pattern
python -m pytest tests/ -v -k "shape"
python -m pytest tests/ -v -k "gradient"
python -m pytest tests/ -v -k "edge"

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Quick config validation
python main.py --cfg job

# Smoke test (quick train with minimal resources)
python main.py mode=train training.batch_size=2 training.n_paths=4 training.horizon=4
```

---

## Error Recovery Guide

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ValueError: HybridBackbone requires non-empty block list` | Empty blocks in config | Add at least one block to backbone config |
| `ValueError: Head latent_size (X) does not match backbone output_dim (Y)` | Dimension mismatch | Set `latent_size` to match `d_model` |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Feature dim mismatch | Align `input_size`, `feature_dim`, and engineer output |
| `ValueError: Feature dimension mismatch` | Engineer `feature_dim` wrong | Fix the `feature_dim` property |
| `ValueError: Temporal ordering violated` | Data leakage in splits | Check cutoff date and window sizes |
| `RuntimeError: Trying to backward through the graph a second time` | Missing `optimizer.zero_grad()` | Zero gradients before backward pass |
| `NaN in loss` | Numerical instability | Check sigma floors, log-return clamping |
| `CUDA out of memory` | Too many paths or large batch | Reduce `n_paths` or `batch_size` |
