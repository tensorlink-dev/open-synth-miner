# CLAUDE.md

## Project Overview

Open Synth Miner is a Hydra-driven research framework for Bittensor SN50 (Synth). It builds hybrid neural architectures that generate 1,000 differentiable price paths for time-series forecasting, with experiment tracking via Weights & Biases and artifact publishing to Hugging Face Hub.

## Quick Reference

```bash
# Install (editable mode required for component auto-discovery)
python -m pip install -e .

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_block_registry.py -v

# Train
python main.py mode=train

# Backtest
python main.py mode=backtest data=default_loader backtest.champion_repo_id=username/SN50-Hybrid-Hub

# View full Hydra config
python main.py --cfg job
```

## Architecture

### Shape Contracts

All models follow strict tensor shape contracts:

- **Backbone**: `(batch, seq_len, input_size)` -> `(batch, latent_size)`
- **Head**: `(batch, latent_size)` -> head-specific parameters
- **SynthModel.forward()**: always returns `(batch, n_paths, horizon)`
- **CRPS input**: `(batch, horizon, n_paths)` via `prepare_paths_for_crps()`

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/models/registry.py` | Component/block/hybrid registries + `discover_components()` auto-discovery |
| `src/models/factory.py` | `SynthModel`, `HybridBackbone`, `ParallelFusion`, simulation functions |
| `src/models/heads.py` | `GBMHead`, `SDEHead`, `HorizonHead`, `SimpleHorizonHead`, `NeuralBridgeHead` |
| `src/models/components/` | Reusable blocks (RNN, GRU, ResConv, BiTCN, Transformer, etc.) |
| `src/data/market_data_loader.py` | Leak-safe market data loader with pluggable sources and feature engineering |
| `src/research/trainer.py` | Training loop + `DataToModelAdapter` |
| `src/research/metrics.py` | CRPS + `CRPSMultiIntervalScorer` with adaptive intervals |
| `src/research/backtest.py` | Champion vs. challenger walk-forward backtesting |
| `src/tracking/hub_manager.py` | HF Hub upload + W&B bridge |
| `configs/` | Hydra YAML configs for models, data, and experiments |

### Adding New Blocks

1. Create the block class in `src/models/components/`
2. Decorate with `@registry.register_block("name")` (auto-discovered at runtime)
3. Add a YAML recipe in `configs/model/` using `_target_` entries
4. No manual imports needed -- `discover_components("src/models/components")` handles it

### Hydra Configuration

Models are instantiated via Hydra `_target_` entries. The root config is `configs/config.yaml` with defaults for model (`configs/model/`) and data (`configs/data/`). Override any value on the command line: `python main.py training.batch_size=8 training.horizon=12`.

## Code Conventions

- Python 3.10+, full type hints, `from __future__ import annotations`
- NumPy-style docstrings
- `PascalCase` classes, `snake_case` functions, `SCREAMING_SNAKE_CASE` constants
- Clamp log-returns with `MAX_LOG_RETURN_CLAMP = 20.0` to prevent `exp()` overflow
- Volatility floor of `1e-6` on all head outputs
- Raise on programmer errors; degrade gracefully on boundary conditions
- Every behavioral change should include shape contract tests and edge case tests

## Testing

Tests live in `tests/` and mirror `src/` structure. Run with `pytest tests/`. Key test areas:

- **Shape contracts**: verify output dimensions match documented contracts
- **Edge cases**: boundary conditions (zero, negative, extreme values)
- **Numerical stability**: extreme inputs that might cause NaN/Inf
- **Integration**: component interactions, not just units

## Environment Variables

- `WANDB_API_KEY` -- required for W&B experiment tracking
- `HF_TOKEN` -- required for Hugging Face Hub artifact uploads
