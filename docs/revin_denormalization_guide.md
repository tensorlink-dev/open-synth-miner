# RevIN Denormalization Guide

## Problem Statement

When using **RevIN (Reversible Instance Normalization)** in your model backbone, the input features are normalized to have zero mean and unit variance. This helps the model learn more effectively from non-stationary time series data. However, **the normalization must be reversed** to get predictions in the original scale.

Previously, the RevIN layer only performed normalization but never denormalized the outputs, leading to predictions in the wrong scale.

## Solution

The `SynthModel` class now:

1. **Automatically detects RevIN layers** in the backbone during initialization
2. **Stores normalization statistics** (mean and std) during forward passes
3. **Denormalizes outputs** by scaling drift (mu) and volatility (sigma) back to the original scale

## Usage

### Basic Usage (Default Behavior)

By default, denormalization is **automatically applied** when RevIN layers are present:

```python
import torch
from src.models.components.advanced_blocks import RevIN, TransformerBlock
from src.models.factory import HybridBackbone, SynthModel
from src.models.heads import GBMHead

# Create model with RevIN in the backbone
blocks = [
    RevIN(d_model=32, affine=True),
    TransformerBlock(d_model=32, nhead=4),
]

backbone = HybridBackbone(
    input_size=5,
    d_model=32,
    blocks=blocks,
)

head = GBMHead(latent_size=32)
model = SynthModel(backbone=backbone, head=head)

# During inference - denormalization is applied by default
model.eval()
with torch.no_grad():
    paths, mu, sigma = model(
        history,
        initial_price=initial_price,
        horizon=288,
        n_paths=1000,
        # apply_revin_denorm=True  # This is the default
    )
```

### Disabling Denormalization

If you want to get outputs in normalized space (e.g., for debugging or specific analysis), set `apply_revin_denorm=False`:

```python
with torch.no_grad():
    paths_normalized, mu_norm, sigma_norm = model(
        history,
        initial_price=initial_price,
        horizon=288,
        n_paths=1000,
        apply_revin_denorm=False,  # Disable denormalization
    )
```

## How It Works

### 1. Normalization (Forward Pass)

When your input passes through a RevIN layer:

```python
# RevIN normalizes the input
mean = x.mean(dim=1, keepdim=True)  # Mean across time dimension
std = sqrt(x.var(dim=1, keepdim=True))  # Std across time dimension
x_normalized = (x - mean) / std
```

The `mean` and `std` are stored as buffers in the RevIN layer.

### 2. Model Processing

The normalized features flow through your backbone and head:
- Backbone processes normalized features
- Head outputs drift `mu` and volatility `sigma` (in normalized space)

### 3. Denormalization (Output)

Before simulating paths, the model scales `mu` and `sigma` back to the original scale:

```python
# Get the average std across features as a scale factor
scale = revin.stdev.mean(dim=-1).squeeze(-1)

# Scale parameters back to original space
mu_denorm = mu * scale
sigma_denorm = sigma * scale
```

This ensures that the simulated paths are in the correct scale relative to your original features.

## Example: Comparing With vs Without Denormalization

```python
import torch
import matplotlib.pyplot as plt

# Assume model and data are set up as above
model.eval()

# Generate paths WITH denormalization (default)
with torch.no_grad():
    paths_denorm, mu_denorm, sigma_denorm = model(
        history,
        initial_price=torch.ones(batch_size),
        horizon=100,
        n_paths=1000,
        apply_revin_denorm=True,
    )

# Generate paths WITHOUT denormalization
with torch.no_grad():
    paths_no_denorm, mu_no_denorm, sigma_no_denorm = model(
        history,
        initial_price=torch.ones(batch_size),
        horizon=100,
        n_paths=1000,
        apply_revin_denorm=False,
    )

# Compare
print(f"With denorm - mu: {mu_denorm[0].item():.4f}, sigma: {sigma_denorm[0].item():.4f}")
print(f"Without denorm - mu: {mu_no_denorm[0].item():.4f}, sigma: {sigma_no_denorm[0].item():.4f}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# With denormalization
sample_paths = paths_denorm[0].cpu().numpy()
axes[0].plot(sample_paths.T, alpha=0.1, color='blue')
axes[0].set_title('With Denormalization')
axes[0].set_ylabel('Price Factor')

# Without denormalization
sample_paths_no = paths_no_denorm[0].cpu().numpy()
axes[1].plot(sample_paths_no.T, alpha=0.1, color='red')
axes[1].set_title('Without Denormalization')

plt.tight_layout()
plt.show()
```

## Important Notes

### When RevIN is NOT Used

If your model backbone doesn't contain any RevIN layers, the `apply_revin_denorm` flag has no effect - the model behaves normally.

```python
# Model without RevIN
blocks = [TransformerBlock(d_model=32, nhead=4)]
backbone = HybridBackbone(input_size=5, d_model=32, blocks=blocks)
model = SynthModel(backbone=backbone, head=GBMHead(latent_size=32))

# apply_revin_denorm=True or False makes no difference
paths1, mu1, sigma1 = model(..., apply_revin_denorm=True)
paths2, mu2, sigma2 = model(..., apply_revin_denorm=False)
# paths1 == paths2, mu1 == mu2, sigma1 == sigma2
```

### Multiple RevIN Layers

If you have multiple RevIN layers in your backbone, the model uses the **first one's statistics** for denormalization. This is because:
- The first RevIN sees the raw input features
- Later RevIN layers operate on already-transformed representations
- The input-level normalization is what needs to be reversed

### Training vs Evaluation

- During **training**, denormalization is still applied by default
- During **evaluation** (`model.eval()`), denormalization ensures predictions are in the correct scale
- You can control this behavior with the `apply_revin_denorm` parameter in both modes

## Migration Guide

If you have existing code that uses RevIN, **no changes are required** - denormalization is now automatic by default. However, you should verify that your predictions are now in the expected scale:

```python
# Old code (predictions were in normalized space)
paths, mu, sigma = model(history, initial_price, horizon, n_paths)

# New code (same call, but now automatically denormalized)
paths, mu, sigma = model(history, initial_price, horizon, n_paths)

# If you need the old behavior (normalized outputs), explicitly disable
paths_norm, mu_norm, sigma_norm = model(
    history, initial_price, horizon, n_paths,
    apply_revin_denorm=False
)
```

## Troubleshooting

### Paths seem too large/small

Check if `apply_revin_denorm=True` (default). If your input features have very large or small scale, the denormalization will amplify or reduce the predictions accordingly.

### Predictions don't match targets

Ensure your target data is in the same scale as your denormalized predictions:
- If targets are price factors (cumulative returns), predictions should be too
- If targets are in log-return space, you may need to adjust comparison

### Performance degradation

Denormalization is a lightweight operation (just multiplication) and should not impact performance. If you see slowdowns, it's likely unrelated to denormalization.

## Technical Details

### Scaling Factor Computation

The denormalization uses the **average standard deviation** across all features:

```python
scale = revin.stdev.mean(dim=-1).squeeze(-1)
```

This gives a single scale factor per batch element. For multi-dimensional features, this averages the per-feature standard deviations to get a representative scale.

### Why Scale Both Mu and Sigma?

- **Mu (drift)**: Expected return per time step. If input features are scaled by `s`, the learned drift is in units of `1/s`, so we multiply by `s` to get original units.
- **Sigma (volatility)**: Standard deviation of returns. Similarly scaled by `s`.

Both parameters are rates of change and thus inversely proportional to the normalization scale.

### Alternative: Denormalize Paths Directly

For `NeuralSDEHead`, which generates paths internally, we denormalize the log-returns:

```python
log_returns = log(paths / initial_price)
log_returns_scaled = log_returns * scale
paths_denorm = initial_price * exp(log_returns_scaled)
```

This achieves the same effect as scaling mu and sigma before simulation.

## References

- **RevIN Paper**: "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift" (Kim et al., 2022)
- **Implementation**: `src/models/components/advanced_blocks.py` (lines 254-283)
- **Denormalization Logic**: `src/models/factory.py` (SynthModel class)
