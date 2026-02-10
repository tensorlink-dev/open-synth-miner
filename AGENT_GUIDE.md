# Agent Research Guide

## TL;DR

```python
from src.research.agent_api import ResearchSession

session = ResearchSession()
result = session.run_preset("transformer_lstm")
print(result["metrics"]["crps"])  # lower is better
```

## What This Project Does

Open Synth Miner builds **probabilistic price forecasting models** for Bittensor SN50. Models take price history as input and output thousands of simulated future price paths. Quality is measured by **CRPS** (Continuous Ranked Probability Score) -- **lower CRPS = better calibrated forecasts**.

The architecture is modular:
- **Backbone blocks** process the input sequence (Transformer, LSTM, Conv, Fourier, etc.)
- **Simulation heads** convert backbone output to stochastic price paths (GBM, SDE, etc.)
- Blocks are composable -- stack any combination in any order

## Research API

### 1. Start a Session

```python
from src.research.agent_api import ResearchSession

session = ResearchSession()
```

No configuration needed. No W&B. No HF Hub. No side effects.

### 2. Discover Components

```python
# What blocks can I use?
blocks = session.list_blocks()
# Returns: [{"name": "TransformerBlock", "description": "...", "params": {...}, "cost": "medium"}, ...]

# What heads can I use?
heads = session.list_heads()
# Returns: [{"name": "GBMHead", "description": "...", "output": "..."}, ...]

# What presets are ready to run?
presets = session.list_presets()
# Returns: [{"name": "transformer_lstm", "description": "...", "tags": [...], "blocks": [...], "head": "..."}, ...]
```

### 3. Run Experiments

**Option A: Run a preset (easiest)**
```python
result = session.run_preset("transformer_lstm")
result = session.run_preset("dlinear_simple")
result = session.run_preset("fourier_lstm")
```

**Option B: Build a custom experiment**
```python
experiment = session.create_experiment(
    blocks=["TransformerBlock", "LSTMBlock"],
    head="GBMHead",
    d_model=32,
    horizon=12,
)
result = session.run(experiment)
```

**Option C: One-liner**
```python
from src.research.agent_api import quick_experiment
result = quick_experiment(blocks=["FourierBlock", "GRUBlock"], d_model=64)
```

### 4. Read Results

Every result is a plain dict:
```python
result = {
    "name": "transformer_lstm",
    "status": "ok",                    # or "error"
    "metrics": {
        "crps": 0.123,                 # LOWER is better
        "sharpness": 0.456,            # ensemble spread
        "log_likelihood": -1.23,       # HIGHER is better
        "final_train_loss": 0.234,
    },
    "param_count": {"total": 12345, "trainable": 12345},
    "duration_seconds": 1.23,
    "epochs": 1,
    "config_summary": {
        "blocks": ["TransformerBlock", "LSTMBlock"],
        "head": "GBMHead",
        "d_model": 32,
        "horizon": 12,
    },
}
```

### 5. Compare Experiments

```python
# Run several experiments
session.run_preset("transformer_lstm")
session.run_preset("dlinear_simple")
session.run_preset("conv_gru")

# Compare them (ranked by CRPS, lower is better)
comparison = session.compare()
print(comparison["best"]["name"])      # best experiment name
print(comparison["ranking"])           # all results, sorted
```

### 6. Sweep All Presets

```python
# Run every preset and compare
comparison = session.sweep()
print(comparison["best"])

# Or specific presets
comparison = session.sweep(["transformer_lstm", "dlinear_simple", "fourier_lstm"])
```

### 7. Validate Before Running

```python
experiment = session.create_experiment(
    blocks=["TransformerBlock", "LSTMBlock"],
    head="GBMHead",
)
validation = session.validate(experiment)
# {"valid": True, "param_count": {...}, "errors": [], "warnings": []}

description = session.describe(experiment)
# Full architecture description without running
```

## Available Presets

| Name | Blocks | Head | Description |
|------|--------|------|-------------|
| `transformer_lstm` | Transformer + LSTM | GBM | Default hybrid (baseline) |
| `pure_transformer` | Transformer x2 | GBM | Attention-only |
| `conv_gru` | ResConv + GRU | GBM | Local patterns + sequence |
| `dlinear_simple` | DLinear | GBM | Lightweight decomposition |
| `fourier_lstm` | Fourier + LSTM | GBM | Spectral + temporal |
| `timesnet` | TimesNet | GBM | Period-aware 2D modeling |
| `timemixer` | TimeMixer | GBM | Multi-scale mixing |
| `transformer_sde_head` | Transformer | SDE | Deeper parameter network |
| `deep_hybrid` | RevIN + Transformer + ResConv + LSTM | GBM | 4-block deep stack |
| `unet_transformer` | U-Net + Transformer | GBM | Multi-resolution attention |

## Available Blocks

| Block | Cost | Best For |
|-------|------|----------|
| `TransformerBlock` | medium | Long-range dependencies, global patterns |
| `LSTMBlock` | medium | Sequential patterns, momentum, trends |
| `GRUBlock` | low-medium | Lighter LSTM alternative |
| `RNNBlock` | low | Minimal recurrent block |
| `ResConvBlock` | low | Local feature extraction |
| `BiTCNBlock` | low | Multi-scale local features via dilation |
| `FourierBlock` | medium | Periodic/seasonal patterns |
| `DLinearBlock` | very low | Trend-seasonal decomposition baseline |
| `TimesNetBlock` | high | Auto period discovery + 2D conv |
| `TimeMixerBlock` | medium | Multi-scale decomposition mixing |
| `RevIN` | very low | Distribution shift (use as first block) |
| `Unet1DBlock` | medium | Multi-resolution features |
| `LayerNormBlock` | very low | Stabilize training between blocks |
| `SDEEvolutionBlock` | low | Stochastic residual updates |
| `TransformerEncoder` | high | Deep multi-layer self-attention |

## Available Heads

| Head | Expressiveness | Description |
|------|---------------|-------------|
| `GBMHead` | Low | Constant drift/vol. Simple, fast, good default. |
| `SDEHead` | Medium | Deeper drift/vol network. |
| `NeuralSDEHead` | Very High | Full neural SDE (state-dependent dynamics). |
| `HorizonHead` | High | Per-step drift/vol via cross-attention. |
| `SimpleHorizonHead` | Medium | Per-step drift/vol via pooling (memory efficient). |
| `NeuralBridgeHead` | High | Hierarchical macro + micro texture paths. |

## Research Strategies for Agents

### Strategy 1: Quick Screening
Run all presets with 1 epoch to find promising architectures:
```python
session = ResearchSession()
comparison = session.sweep(epochs=1)
top_3 = comparison["ranking"][:3]
```

### Strategy 2: Architecture Search
Systematically vary blocks and heads:
```python
session = ResearchSession()
for head in ["GBMHead", "SDEHead"]:
    for blocks in [
        ["TransformerBlock", "LSTMBlock"],
        ["FourierBlock", "GRUBlock"],
        ["DLinearBlock"],
    ]:
        exp = session.create_experiment(blocks=blocks, head=head)
        session.run(exp)
comparison = session.compare()
```

### Strategy 3: Hyperparameter Search
Vary d_model, horizon, learning rate:
```python
session = ResearchSession()
for d_model in [16, 32, 64]:
    for lr in [0.01, 0.001, 0.0001]:
        exp = session.create_experiment(
            blocks=["TransformerBlock", "LSTMBlock"],
            d_model=d_model,
            lr=lr,
        )
        session.run(exp, name=f"d{d_model}_lr{lr}")
comparison = session.compare()
```

### Strategy 4: Ablation Study
Test the contribution of each block:
```python
session = ResearchSession()
full = session.create_experiment(
    blocks=["RevIN", "TransformerBlock", "ResConvBlock", "LSTMBlock"],
)
session.run(full, name="full_model")

# Remove one block at a time
for i, block_name in enumerate(["RevIN", "TransformerBlock", "ResConvBlock", "LSTMBlock"]):
    remaining = ["RevIN", "TransformerBlock", "ResConvBlock", "LSTMBlock"]
    remaining.pop(i)
    exp = session.create_experiment(blocks=remaining)
    session.run(exp, name=f"without_{block_name}")
comparison = session.compare()
```

## Key Metric: CRPS

**CRPS (Continuous Ranked Probability Score)** measures how well the ensemble of simulated paths matches reality. It rewards:
- **Calibration**: paths that bracket the actual outcome
- **Sharpness**: tight, confident distributions (not overly wide)

**Lower CRPS = better model.**

## Error Handling

Results always have a `status` field:
```python
if result["status"] == "ok":
    print(result["metrics"]["crps"])
elif result["status"] == "error":
    print(result["error"])      # error message
    print(result["traceback"])  # full traceback
```

Experiments never raise exceptions -- errors are captured in the result dict.
