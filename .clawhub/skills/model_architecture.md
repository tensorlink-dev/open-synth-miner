# Skill: Model Architecture

## Purpose
Design, register, and compose hybrid neural architectures for stochastic price path generation within the Open Synth Miner framework.

## When to Use
- User asks to create a new model block, head, or backbone configuration
- User wants to modify an existing architecture or compose blocks differently
- User needs to understand the model pipeline or debug shape issues

---

## Core Concepts

### The Model Pipeline

```
Input: (batch, seq_len, feature_dim)
    │
    ▼
┌─────────────────────────────────┐
│ HybridBackbone                   │
│  input_proj: Linear(feature_dim → d_model)
│  blocks: [Block₁, Block₂, ...]  │  ← registered blocks from YAML
│  output: last timestep           │
│  Shape: (batch, d_model)         │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Head (HeadBase subclass)         │
│  Maps latent → stochastic params │
│  Type determines simulation fn   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Simulation                       │
│  GBM / Horizon / Bridge / etc.   │
│  Output: (batch, n_paths, horizon)│
└─────────────────────────────────┘
```

### Key Classes

| Class | Location | Role |
|-------|----------|------|
| `SynthModel` | `src/models/factory.py` | Top-level model wrapping backbone + head |
| `HybridBackbone` | `src/models/factory.py` | Composes registered blocks into a backbone |
| `HeadBase` | `src/models/heads.py` | Abstract base for all stochastic heads |
| `BlockBase` | `src/models/backbones.py` | Abstract base for backbone blocks |
| `Registry` | `src/models/registry.py` | Decorator-based component registry |
| `ParallelFusion` | `src/models/factory.py` | Run multiple blocks in parallel and fuse |

---

## Task: Register a New Block

### Step-by-step

1. **Create the file**: `src/models/components/<name>.py`

2. **Implement the block**:
```python
import torch
import torch.nn as nn
from src.models.registry import registry


@registry.register_block(
    "myblock",
    description="Description of what this block does",
    preserves_seq_len=True,    # Does it keep sequence length?
    preserves_d_model=True,    # Does it keep feature dimension?
    min_seq_len=1,             # Minimum input sequence length
)
class MyBlock(nn.Module):
    """One-line description.

    Parameters
    ----------
    d_model : int
        Feature dimension (must match backbone d_model).
    """

    def __init__(self, d_model: int, **kwargs) -> None:
        super().__init__()
        # Build layers here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input tensor.

        Parameters
        ----------
        x : (batch, seq_len, d_model)

        Returns
        -------
        torch.Tensor : (batch, seq_len, d_model)
        """
        # Implementation here
        return x
```

3. **No registration call needed** — `discover_components("src/models/components")` auto-imports all `.py` files

4. **Use in config** (`configs/model/my_recipe.yaml`):
```yaml
model:
  _target_: src.models.factory.SynthModel
  backbone:
    _target_: src.models.factory.HybridBackbone
    input_size: ${data.feature_dim}
    d_model: 32
    blocks:
      - _target_: src.models.registry.MyBlock
        d_model: 32
      - _target_: src.models.registry.LSTMBlock
        d_model: 32
  head:
    _target_: src.models.heads.GBMHead
    latent_size: 32
```

5. **Write tests** in `tests/test_<name>.py`:
```python
import torch
from src.models.components.<name> import MyBlock

def test_myblock_shape_contract():
    block = MyBlock(d_model=32)
    x = torch.randn(2, 16, 32)
    out = block(x)
    assert out.shape == (2, 16, 32)

def test_myblock_gradient_flow():
    block = MyBlock(d_model=32)
    x = torch.randn(2, 16, 32, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None
```

### Validation Checklist
- [ ] Block accepts `d_model` as first positional arg
- [ ] `forward()` preserves `(batch, seq, d_model)` shape (or declares otherwise)
- [ ] Gradients flow through the block
- [ ] No in-place operations that break autograd
- [ ] Numerical stability: no unbounded `exp()`, `log()`, or division
- [ ] Tests pass: `python -m pytest tests/test_<name>.py -v`

---

## Task: Create a New Head

### Step-by-step

1. **Subclass HeadBase** in `src/models/heads.py`:
```python
class MyHead(HeadBase):
    """Describe what parameters this head predicts.

    Parameters
    ----------
    latent_size : int
        Must match backbone d_model / output_dim.
    """

    def __init__(self, latent_size: int) -> None:
        super().__init__()
        self.mu_proj = nn.Linear(latent_size, 1)
        self.sigma_proj = nn.Linear(latent_size, 1)

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_proj(h_t).squeeze(-1)
        sigma = F.softplus(self.sigma_proj(h_t)).squeeze(-1) + 1e-6  # volatility floor
        return mu, sigma
```

2. **Register in HEAD_REGISTRY** (`src/models/factory.py`):
```python
HEAD_REGISTRY = {
    ...
    "my_head": MyHead,
}
```

3. **Add routing in SynthModel.forward()** (`src/models/factory.py`):
   - If your head returns `(mu, sigma)` scalars, it falls through to the default GBM simulation — no changes needed.
   - If it returns custom parameters, add an `isinstance` check before the else clause.

4. **Critical rules for heads**:
   - Always add `+ 1e-6` to sigma for numerical stability
   - Use `F.softplus()` for positive outputs, `torch.tanh()` for bounded
   - Return must be documented: what shapes, what semantics
   - `latent_size` must match backbone `output_dim`

---

## Task: Compose an Architecture via Config

### Hydra YAML Pattern

```yaml
# configs/model/my_architecture.yaml
model:
  _target_: src.models.factory.SynthModel
  backbone:
    _target_: src.models.factory.HybridBackbone
    input_size: ${data.feature_dim}    # Resolved from data config
    d_model: 64                         # Internal feature dimension
    validate_shapes: true               # Shape validation at construction
    insert_layernorm: false             # Auto-insert LayerNorm between blocks
    blocks:
      - _target_: src.models.registry.TransformerBlock
        d_model: 64
        nhead: 8
        dropout: 0.1
      - _target_: src.models.registry.LSTMBlock
        d_model: 64
        num_layers: 2
  head:
    _target_: src.models.heads.HorizonHead
    latent_size: 64
    horizon_max: 48
    nhead: 4
    n_layers: 2
```

### Parallel Fusion Pattern

```yaml
backbone:
  _target_: src.models.factory.HybridBackbone
  input_size: 3
  d_model: 32
  blocks:
    - _target_: src.models.factory.ParallelFusion
      merge_strategy: gating    # or "concat"
      paths:
        - _target_: src.models.registry.TransformerBlock
          d_model: 32
        - _target_: src.models.registry.LSTMBlock
          d_model: 32
```

---

## Available Blocks Reference

| Block | Key Params | Shape Change | Description |
|-------|-----------|--------------|-------------|
| `TransformerBlock` | `d_model, nhead, dropout` | Preserves | Self-attention + gated MLP |
| `LSTMBlock` | `d_model, num_layers, dropout` | Preserves | LSTM sequence modeling |
| `SDEEvolutionBlock` | `d_model, hidden, dropout` | Preserves | Residual stochastic updates |
| `RNNBlock` | `d_model, num_layers` | Preserves | Elman RNN |
| `GRUBlock` | `d_model, num_layers` | Preserves | GRU sequence modeling |
| `ResConvBlock` | `d_model, kernel_size` | Preserves | 1D residual convolution |
| `BiTCNBlock` | `d_model, kernel_size, num_layers` | Preserves | Bidirectional temporal conv |
| `LayerNormBlock` | `d_model` | Preserves | Standalone LayerNorm |
| `PatchMerging` | `d_model` | Changes seq_len | Avg-pool downsampling |

## Available Heads Reference

| Head | Params | Returns | Simulation |
|------|--------|---------|------------|
| `GBMHead` | `latent_size` | `(mu, sigma)` | `simulate_gbm_paths` |
| `SDEHead` | `latent_size, hidden` | `(mu, sigma)` | `simulate_gbm_paths` |
| `NeuralSDEHead` | `latent_size, hidden, solver, adjoint` | `(paths, mu, sigma)` | Internal torchsde |
| `HorizonHead` | `latent_size, horizon_max, nhead, n_layers, kv_dim` | `(mu_seq, sigma_seq)` | `simulate_horizon_paths` |
| `SimpleHorizonHead` | `latent_size, horizon_max, pool_type` | `(mu_seq, sigma_seq)` | `simulate_horizon_paths` |
| `MixtureDensityHead` | `latent_size, n_components` | `(mus, sigmas, weights)` | `simulate_mixture_paths` |
| `VolTermStructureHead` | `latent_size, hidden` | `(mu_seq, sigma_seq)` | `simulate_horizon_paths` |
| `NeuralBridgeHead` | `latent_size, micro_steps, hidden_dim` | `(macro_ret, micro_returns, sigma)` | `simulate_bridge_paths` |

---

## Common Pitfalls

1. **d_model mismatch**: Every block in a backbone must use the same `d_model`. The backbone's `input_proj` handles `feature_dim → d_model`.
2. **latent_size mismatch**: Head's `latent_size` must equal backbone's `output_dim` (which equals `d_model` for standard blocks).
3. **Forgetting volatility floor**: Always `sigma = F.softplus(...) + 1e-6`.
4. **Breaking gradient flow**: Don't use `.detach()` or `.item()` inside forward passes.
5. **Shape validation disabled**: If `validate_shapes=false`, shape bugs surface at training time instead of construction time.
