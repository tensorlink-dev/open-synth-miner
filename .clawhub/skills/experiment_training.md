# Skill: Experiment & Training

## Purpose
Train SynthModel architectures, compute CRPS loss, and orchestrate experiment runs with W&B logging.

## When to Use
- User wants to train a model from scratch or fine-tune
- User needs to set up a training loop with the leak-safe data loader
- User asks about loss functions, optimizers, or training configuration
- User wants to run experiments with W&B tracking

---

## Core Concepts

### Training Pipeline

```
Hydra Config
    │
    ▼
run_experiment(cfg)
    ├── create_model(cfg)           ← Instantiate SynthModel from config
    ├── wandb.init(project, config) ← Start W&B run
    ├── train_step(model, batch, optimizer, horizon, n_paths)
    │       ├── model.forward(history, initial_price, horizon, n_paths)
    │       │       → (paths, mu, sigma)  shape: (batch, n_paths, horizon)
    │       ├── crps_ensemble(terminal_paths, target)  ← loss
    │       ├── loss.backward()
    │       └── optimizer.step()
    └── evaluate_and_log(model, batch, horizon, n_paths, step)
            ├── CRPS, sharpness, log-likelihood
            ├── Multi-interval CRPS (if actual_series available)
            └── log_experiment_results → W&B
```

### Key Classes & Functions

| Entity | Location | Role |
|--------|----------|------|
| `run_experiment(cfg)` | `src/research/experiment_mgr.py` | Full experiment orchestrator |
| `train_step(model, batch, optimizer, ...)` | `src/research/trainer.py` | Single training step (legacy API) |
| `evaluate_and_log(model, batch, ...)` | `src/research/trainer.py` | Eval + W&B logging |
| `Trainer` | `src/research/trainer.py` | OOP trainer with DataToModelAdapter |
| `DataToModelAdapter` | `src/research/trainer.py` | Bridges loader format → model format |
| `prepare_paths_for_crps(paths)` | `src/research/trainer.py` | Transpose: `(B, P, H) → (B, H, P)` |
| `crps_ensemble(simulations, target)` | `src/research/metrics.py` | Vectorized CRPS loss |
| `afcrps_ensemble(simulations, target, alpha)` | `src/research/metrics.py` | Almost-fair CRPS (default alpha=0.95) |
| `log_likelihood(simulations, target)` | `src/research/metrics.py` | Gaussian log-likelihood estimate |

---

## Task: Run a Quick Experiment via CLI

```bash
# Default config: TransformerBlock → LSTMBlock → GBMHead
python main.py mode=train

# Override parameters
python main.py mode=train \
    training.batch_size=8 \
    training.seq_len=64 \
    training.horizon=12 \
    training.n_paths=256 \
    training.lr=0.0005

# Use a different model recipe
python main.py mode=train model=hybrid_with_layernorm

# Use real OHLCV data
python main.py mode=train data=ohlcv_loader
```

---

## Task: Custom Training Loop with Leak-Safe Data

Use the `Trainer` class for full control over training with the `MarketDataLoader`:

```python
import torch
import torch.optim as optim
from src.models.factory import create_model
from src.models.registry import discover_components
from src.data import MockDataSource, ZScoreEngineer, MarketDataLoader
from src.research.trainer import Trainer, DataToModelAdapter
import pandas as pd

# 1. Setup
discover_components("src/models/components")

# 2. Build model from config
from omegaconf import OmegaConf
cfg = OmegaConf.create({
    "model": {
        "_target_": "src.models.factory.SynthModel",
        "backbone": {
            "_target_": "src.models.factory.HybridBackbone",
            "input_size": 3,
            "d_model": 32,
            "blocks": [
                {"_target_": "src.models.registry.TransformerBlock", "d_model": 32, "nhead": 4},
                {"_target_": "src.models.registry.LSTMBlock", "d_model": 32},
            ],
        },
        "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 32},
    },
    "training": {"horizon": 12, "n_paths": 256, "feature_dim": 3},
})
model = create_model(cfg)

# 3. Build data pipeline
source = MockDataSource(length=5000, freq="5min")
engineer = ZScoreEngineer()
loader = MarketDataLoader(
    data_source=source, engineer=engineer,
    assets=["BTC"], input_len=96, pred_len=12, batch_size=16,
)
train_dl, val_dl, test_dl = loader.static_holdout(
    pd.Timestamp("2020-01-10", tz="UTC")
)

# 4. Create trainer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    n_paths=256,
    crps_alpha=0.95,  # Almost-fair CRPS
)

# 5. Training loop
for epoch in range(10):
    epoch_loss = 0.0
    for batch in train_dl:
        metrics = trainer.train_step(batch)
        epoch_loss += metrics["loss"]
    avg_loss = epoch_loss / len(train_dl)

    # Validate
    val_metrics = trainer.validate(val_dl)
    print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_crps={val_metrics['val_crps']:.4f}")
```

---

## Task: Use run_experiment() for W&B-Tracked Runs

```python
from src.research.experiment_mgr import run_experiment
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/config.yaml")
# Override as needed
cfg.training.batch_size = 8
cfg.training.n_paths = 512

result = run_experiment(cfg)
# result contains: model, metrics, config, run (wandb), recipe, block_hash
print(f"CRPS: {result['metrics']['crps']:.6f}")
```

---

## Loss Functions

### CRPS (Continuous Ranked Probability Score)
The primary loss for probabilistic forecasts. Measures how well the ensemble distribution matches the observation.

```python
from src.research.metrics import crps_ensemble, afcrps_ensemble

# Standard CRPS
loss = crps_ensemble(simulations, target)  # (batch,)

# Almost-fair CRPS (recommended, alpha=0.95)
# Removes finite-ensemble bias
loss = afcrps_ensemble(simulations, target, alpha=0.95)  # (batch,)
```

**Shape expectations**:
- `simulations`: `(batch, n_ensemble_members)` — ensemble in last dimension
- `target`: `(batch,)` — single observation per sample
- Returns: `(batch,)` — per-sample CRPS

### Log-Likelihood
Secondary metric for diagnostic monitoring:
```python
from src.research.metrics import log_likelihood
ll = log_likelihood(simulations, target)  # (batch,)
```

### Multi-Interval CRPS
For backtesting: evaluates at 5min, 30min, 3hour, and 24hour intervals:
```python
from src.research.metrics import CRPSMultiIntervalScorer

scorer = CRPSMultiIntervalScorer(
    time_increment=60,    # seconds per timestep
    adaptive=True,        # Auto-adapt intervals to horizon
)
total_crps, detail = scorer(simulation_paths, actual_price_path)
```

---

## DataToModelAdapter

The `DataToModelAdapter` bridges the gap between `MarketDataLoader` batch format and `SynthModel` input format:

| Loader Output | Model Expects | Adapter Action |
|--------------|--------------|----------------|
| `inputs: (B, F, T)` | `history: (B, T, F)` | Transpose |
| `target: (B, 1, pred_len)` | `target_factors: (B, pred_len)` | Squeeze + cumsum + exp |
| — | `initial_price: (B,)` | Always `1.0` (relative factors) |

When the adapter sets `initial_price = 1.0`, the model's output paths become **price factors** rather than absolute prices. CRPS is then computed between simulated factors and actual factors.

---

## Training Configuration Reference

```yaml
training:
  batch_size: 4          # Samples per gradient step
  seq_len: 32            # Input sequence length
  feature_dim: 3         # Must match engineer.feature_dim
  horizon: 12            # Prediction steps
  n_paths: 1000          # Monte Carlo paths per sample
  lr: 0.001              # Learning rate
  initial_price: 100.0   # For dummy batch in run_experiment
  target_std: 1.0        # For dummy batch noise
```

---

## Common Pitfalls

1. **Feature dim mismatch**: `training.feature_dim` must match `data.feature_dim` must match `model.backbone.input_size`. The data config should define `feature_dim` and the root config resolves `${data.feature_dim}`.

2. **Forgetting `prepare_paths_for_crps()`**: Model returns `(B, P, H)`, CRPS expects `(B, H, P)`. Always transpose before computing loss.

3. **Not detaching targets**: Use `target = batch["target"].detach()` to prevent gradient flow through the target.

4. **Gradient accumulation**: `optimizer.zero_grad()` must be called before each `loss.backward()`.

5. **Using `.item()` too early**: Don't call `.item()` on tensors that need gradients. Only use after `loss.backward()`.
