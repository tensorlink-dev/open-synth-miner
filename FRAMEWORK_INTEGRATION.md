# Synth Research Agent Framework — Dependency Context

This repo builds a lightweight agent framework on top of `open-synth-miner`,
a PyTorch research framework for probabilistic price forecasting on Bittensor SN50.

This document describes the full integration surface of `open-synth-miner` —
every function, dict shape, tensor contract, and config schema your framework
can use. Treat this as the ground-truth API reference.

---

## 1. What open-synth-miner does

Models take price history → produce 1000s of Monte Carlo future price paths.
Quality is measured by **CRPS** (lower = better calibrated probabilistic forecasts).

Architecture is two-part:
- **Backbone**: stack of composable blocks processing `(batch, seq, features)` → `(batch, latent)`
- **Head**: converts latent → stochastic parameters → simulated paths `(batch, n_paths, horizon)`

The framework provides a zero-side-effect research API (`ResearchSession`) plus
production tracking (W&B + HF Hub) when the agent is ready to publish.

---

## 2. Installation

```
pip install -e /path/to/open-synth-miner
# or as dependency:
# open-synth-miner @ git+https://github.com/tensorlink-dev/open-synth-miner.git
```

**Runtime dependencies**: torch>=2.2, hydra-core>=1.3.2, omegaconf>=2.3, numpy, pandas, properscoring, wandb, huggingface_hub, torchsde

---

## 3. Primary API: ResearchSession

This is the main interface your framework should wrap. All I/O is plain Python dicts.
No side effects (no W&B, no HF Hub, no file writes) unless you opt in separately.

```python
from src.research.agent_api import ResearchSession

session = ResearchSession()
```

### 3.1 Discovery methods

```python
session.list_blocks() -> list[dict]
```
Each entry:
```python
{
    "name": str,           # e.g. "TransformerBlock"
    "target": str,         # e.g. "src.models.registry.TransformerBlock"
    "description": str,    # human-readable
    "params": dict,        # {"d_model": "int (model dimension)", "nhead": "int (default 4)", ...}
    "strengths": str,      # e.g. "Captures long-range dependencies"
    "cost": str,           # "very low" | "low" | "low-medium" | "medium" | "high"
}
```

```python
session.list_heads() -> list[dict]
```
Each entry:
```python
{
    "name": str,           # e.g. "GBMHead"
    "target": str,         # e.g. "src.models.heads.GBMHead"
    "description": str,
    "params": dict,        # {"latent_size": "int (auto-set from backbone)", ...}
    "strengths": str,
    "output": str,         # e.g. "(mu, sigma) scalars -> simulated price paths"
}
```

```python
session.list_presets() -> list[dict]
```
Each entry:
```python
{
    "name": str,           # e.g. "transformer_lstm"
    "description": str,
    "tags": list[str],     # e.g. ["baseline", "hybrid", "recurrent"]
    "blocks": list[str],   # e.g. ["TransformerBlock", "LSTMBlock"]
    "head": str,           # e.g. "GBMHead"
}
```

### 3.2 Experiment construction

```python
session.create_experiment(
    blocks: list[str | dict],   # block names or raw _target_ dicts
    head: str = "GBMHead",      # head name
    *,
    d_model: int = 32,          # hidden dimension
    feature_dim: int = 4,       # input features
    seq_len: int = 32,          # input sequence length
    horizon: int = 12,          # prediction steps
    n_paths: int = 100,         # Monte Carlo paths
    batch_size: int = 4,
    lr: float = 0.001,
    head_kwargs: dict | None = None,        # extra head params
    block_kwargs: list[dict] | None = None, # per-block extra params
) -> dict  # experiment config
```

Returns:
```python
{
    "name": str,  # auto-generated from block+head names
    "model": {
        "backbone": {
            "_target_": "src.models.factory.HybridBackbone",
            "input_size": int,
            "d_model": int,
            "blocks": [{"_target_": str, "d_model": int, ...}, ...],
        },
        "head": {"_target_": str, "latent_size": int, ...},
    },
    "training": {
        "batch_size": int, "seq_len": int, "feature_dim": int,
        "horizon": int, "n_paths": int, "lr": float,
    },
}
```

### 3.3 Validation (no execution)

```python
session.validate(experiment: dict) -> dict
```
Returns:
```python
{
    "valid": bool,
    "param_count": {"total": int, "trainable": int},
    "errors": list[str],
    "warnings": list[str],
}
```

```python
session.describe(experiment: dict) -> dict
```
Returns:
```python
{
    "name": str,
    "blocks": list[str],
    "head": str,
    "d_model": int,
    "feature_dim": int,
    "param_count": {"total": int, "trainable": int},
    "training": {"horizon": int, "n_paths": int, "batch_size": int, "seq_len": int, "lr": float},
    "validation": dict,  # same as validate() output
}
```

### 3.4 Execution

```python
session.run(experiment: dict, *, epochs: int = 1, name: str | None = None) -> dict
```
Returns on success:
```python
{
    "name": str,
    "status": "ok",
    "metrics": {
        "crps": float,              # PRIMARY METRIC — lower is better
        "sharpness": float,         # ensemble spread
        "log_likelihood": float,    # higher is better
        "final_train_loss": float,
    },
    "training_history": [{"loss": float, "crps": float, "sharpness": float, "log_likelihood": float}, ...],
    "param_count": {"total": int, "trainable": int},
    "config_summary": {
        "blocks": list[str],
        "head": str,
        "d_model": int,
        "horizon": int,
        "n_paths": int,
    },
    "duration_seconds": float,
    "epochs": int,
}
```
Returns on error:
```python
{
    "name": str,
    "status": "error",
    "error": str,
    "traceback": str,
    "duration_seconds": float,
    "epochs": 0,
}
```

**Experiments never raise exceptions.** Errors are always in the result dict.

```python
session.run_preset(preset_name: str, *, epochs: int = 1, overrides: dict | None = None) -> dict
# overrides example: {"training": {"horizon": 24, "lr": 0.0001}}
```

```python
session.sweep(preset_names: list[str] | None = None, *, epochs: int = 1) -> dict
# Runs multiple presets, returns comparison. None = all presets.
```

### 3.5 Comparison

```python
session.compare(results: list[dict] | None = None) -> dict
# None = compare all results in this session
```
Returns:
```python
{
    "ranking": [  # sorted by CRPS ascending (best first)
        {
            "name": str,
            "crps": float,
            "sharpness": float,
            "log_likelihood": float,
            "param_count": int,          # trainable
            "duration_seconds": float,
            "blocks": list[str],
            "head": str,
        },
        ...
    ],
    "best": dict | None,       # first entry in ranking
    "worst": dict | None,      # last entry in ranking
    "num_experiments": int,
    "num_failed": int,
    "failed": [{"name": str, "error": str}, ...],
}
```

### 3.6 Session state

```python
session.results -> list[dict]     # all results accumulated
session.summary() -> dict         # {"num_experiments", "comparison", "all_results"}
session.clear() -> None           # reset
```

### 3.7 One-liner convenience

```python
from src.research.agent_api import quick_experiment

result = quick_experiment(
    blocks=["TransformerBlock", "LSTMBlock"],  # default if omitted
    head="GBMHead",                             # default
    d_model=32,
    horizon=12,
)
```

---

## 4. Production tracking (opt-in side effects)

When the agent decides a model is good enough to publish:

### 4.1 HF Hub publishing

```python
from src.tracking.hub_manager import HubManager
from omegaconf import OmegaConf

manager = HubManager(
    run=wandb_run,                        # wandb.Run object
    backbone_name="HybridBackbone",       # string
    head_name="GBMHead",                  # string
    block_hash="abc123def456",            # from registry.recipe_hash()
    recipe=[...],                         # block config list
    architecture_graph=dict,              # backbone config dict
    resolved_config=dict,                 # full resolved config
    repo_id="username/SN50-Hybrid-Hub",   # HF Hub repo
)

hf_link = manager.save_and_push(model=model, crps_score=0.123)
# Returns: "https://huggingface.co/username/SN50-Hybrid-Hub/tree/main/..."
# Side effects: saves model.pt, README.md, config.yaml, uploads to HF Hub

report = manager.get_shareable_report(crps_score=0.123, hf_link=hf_link)
# Returns: formatted markdown string with model details
```

### 4.2 W&B logging

```python
from src.tracking.wandb_logger import log_experiment_results, log_backtest_results

log_experiment_results(
    metrics={"crps": 0.123, "sharpness": 0.456},  # dict[str, float]
    paths=paths_tensor,     # (batch, n_paths, horizon)
    actual_prices=tensor,   # (batch, horizon)
    horizon=12,
    step=0,
)
# Side effects: logs scalars, fan charts, histograms to active W&B run
```

### 4.3 Typical publish flow

```python
import wandb
from omegaconf import OmegaConf
from src.models.factory import create_model
from src.models.registry import discover_components, registry
from src.tracking.hub_manager import HubManager

# 1. Agent already has a result from session.run()
best_experiment = ...  # the experiment config dict
best_result = ...      # the result dict

# 2. Recreate model for saving
discover_components("src/models/components")
cfg = OmegaConf.create(best_experiment)
model = create_model(cfg)
# (agent would need to retrain or keep model reference)

# 3. Get recipe hash
recipe = best_experiment["model"]["backbone"]["blocks"]
block_hash = registry.recipe_hash(recipe)

# 4. Init W&B + publish
run = wandb.init(project="synth-miner", config=best_experiment)
manager = HubManager(
    run=run,
    backbone_name="HybridBackbone",
    head_name=best_experiment["model"]["head"]["_target_"].split(".")[-1],
    block_hash=block_hash,
    recipe=recipe,
    resolved_config=best_experiment,
)
hf_link = manager.save_and_push(model=model, crps_score=best_result["metrics"]["crps"])
wandb.finish()
```

---

## 5. Lower-level model API (for custom training loops)

If your agent framework needs to go beyond ResearchSession (real data, multi-epoch
training, custom losses):

### 5.1 Model creation

```python
from src.models.factory import create_model, build_model, get_model
from src.models.registry import discover_components

discover_components("src/models/components")  # MUST call before model creation

# From a config dict (Hydra-style with _target_ keys):
model = create_model(config_dict_or_omegaconf)

# From HF Hub:
model = get_model({"model": {"hf_repo_id": "user/repo", "architecture": arch_cfg}})
```

### 5.2 Model forward pass

```python
paths, mu, sigma = model(
    x,                          # (batch, seq_len, features) — price history
    initial_price=price,        # (batch,) — current price level
    horizon=12,                 # prediction steps
    n_paths=1000,               # Monte Carlo paths
    dt=1.0,                     # time step scale
    apply_revin_denorm=True,    # auto-denormalize if RevIN blocks present
)
# paths: (batch, n_paths, horizon) — simulated price trajectories
# mu: (batch,) or scalar — drift estimate
# sigma: (batch,) or scalar — volatility estimate
```

### 5.3 Training with real data

```python
from src.data.market_data_loader import MarketDataLoader
from src.research.trainer import Trainer, DataToModelAdapter

# DataLoader yields batches shaped:
# {"inputs": (B, features, time), "target": (B, 1, horizon), "meta": {...}}

adapter = DataToModelAdapter(device=torch.device("cpu"))
trainer = Trainer(model=model, optimizer=optimizer, n_paths=100, adapter=adapter)

for batch in train_loader:
    metrics = trainer.train_step(batch)
    # {"loss", "crps", "sharpness", "log_likelihood", "mu", "sigma"}

val_metrics = trainer.validate(val_loader)
# {"val_crps": float}
```

### 5.4 CRPS computation

```python
from src.research.metrics import afcrps_ensemble, crps_ensemble, log_likelihood
from src.research.trainer import prepare_paths_for_crps

# Model outputs (batch, n_paths, horizon), CRPS wants (batch, horizon, n_paths)
sim_paths = prepare_paths_for_crps(paths)

crps = afcrps_ensemble(sim_paths, target, alpha=0.95)  # recommended
crps = crps_ensemble(sim_paths, target)                 # standard
loglik = log_likelihood(sim_paths, target)
```

---

## 6. Available components

### 6.1 Blocks (15 registered)

| Name | `_target_` | Cost | Best for |
|------|-----------|------|----------|
| TransformerBlock | src.models.registry.TransformerBlock | medium | Long-range patterns |
| LSTMBlock | src.models.registry.LSTMBlock | medium | Sequential/momentum |
| GRUBlock | src.models.components.advanced_blocks.GRUBlock | low-med | Lighter LSTM |
| RNNBlock | src.models.components.advanced_blocks.RNNBlock | low | Minimal recurrence |
| ResConvBlock | src.models.components.advanced_blocks.ResConvBlock | low | Local features |
| BiTCNBlock | src.models.components.advanced_blocks.BiTCNBlock | low | Dilated local |
| FourierBlock | src.models.components.advanced_blocks.FourierBlock | medium | Periodic patterns |
| DLinearBlock | src.models.components.advanced_blocks.DLinearBlock | very low | Decomposition baseline |
| TimesNetBlock | src.models.components.advanced_blocks.TimesNetBlock | high | Period-aware 2D |
| TimeMixerBlock | src.models.components.advanced_blocks.TimeMixerBlock | medium | Multi-scale mixing |
| RevIN | src.models.components.advanced_blocks.RevIN | very low | Input normalization (FIRST) |
| Unet1DBlock | src.models.components.advanced_blocks.Unet1DBlock | medium | Multi-resolution |
| LayerNormBlock | src.models.components.advanced_blocks.LayerNormBlock | very low | Inter-block norm |
| SDEEvolutionBlock | src.models.registry.SDEEvolutionBlock | low | Stochastic residual |
| TransformerEncoder | src.models.components.advanced_blocks.TransformerEncoderAdapter | high | Deep attention |

All blocks: `(batch, seq, d_model)` → `(batch, seq, d_model)`. Any order.

### 6.2 Heads (6 types)

| Name | `_target_` | Expressiveness |
|------|-----------|---------------|
| GBMHead | src.models.heads.GBMHead | Low — constant μ, σ |
| SDEHead | src.models.heads.SDEHead | Medium — deeper μ, σ network |
| HorizonHead | src.models.heads.HorizonHead | High — per-step via cross-attention |
| SimpleHorizonHead | src.models.heads.SimpleHorizonHead | Medium — per-step via pooling |
| NeuralBridgeHead | src.models.heads.NeuralBridgeHead | High — macro+micro hierarchy |
| NeuralSDEHead | src.models.heads.NeuralSDEHead | Very high — full neural SDE |

### 6.3 Presets (10 ready-to-run)

transformer_lstm, pure_transformer, conv_gru, dlinear_simple, fourier_lstm,
timesnet, timemixer, transformer_sde_head, deep_hybrid, unet_transformer

---

## 7. Composition rules

- `d_model` must be divisible by `nhead` (default nhead=4, so use 16/32/64/128)
- `RevIN` must be the **first** block if used
- `LayerNormBlock` goes **between** other blocks
- Deeper stacks (3-4 blocks) need d_model >= 32
- `latent_size` in heads is auto-set to match backbone d_model
- `input_size` / `feature_dim` must match the data source (default: 4)

---

## 8. Config schema (Hydra)

For agents that need to generate YAML or use the CLI:

```yaml
# Top level
mode: train | backtest
project: str

training:
  batch_size: int
  seq_len: int
  feature_dim: int
  horizon: int
  n_paths: int
  lr: float

model:
  _target_: src.models.factory.SynthModel
  backbone:
    _target_: src.models.factory.HybridBackbone
    input_size: int          # must match feature_dim
    d_model: int
    blocks:
      - _target_: <block target path>
        d_model: int         # must match backbone d_model
        <block-specific params>
  head:
    _target_: <head target path>
    latent_size: int         # must match backbone d_model
    <head-specific params>
```

---

## 9. Key design constraints for the framework

1. **ResearchSession is stateful** — accumulates results. Use `session.clear()` between unrelated batches.
2. **Experiments use synthetic data by default** — for real data training, use the Trainer class directly with MarketDataLoader.
3. **CRPS is the only metric that matters for SN50 ranking** — sharpness and log_likelihood are diagnostics.
4. **n_paths=100 in research mode** for speed; production uses n_paths=1000.
5. **All _target_ paths are real module paths** — Hydra instantiate() resolves them at runtime.
6. **discover_components() must be called once** before any model creation — ResearchSession does this automatically, but raw factory usage requires it.
7. **torch tensors throughout** — no numpy in model I/O. Metrics internally convert as needed.
