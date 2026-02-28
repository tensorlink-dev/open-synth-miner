# Open Synth Miner — Agent Instructions

> **`osa`** stands for **O**pen **S**ynth **A**rchitecture — the short import name and CLI entry point for this package.

This is a Hydra-driven research framework for **Bittensor SN50 (Synth)** focused on hybrid neural architectures for time-series forecasting and stochastic price path simulation.

## Quick Reference

| What | Where |
|------|-------|
| Entry point | `main.py` (Hydra CLI: `mode=train` or `mode=backtest`) |
| Model factory | `src/models/factory.py` → `create_model(cfg)`, `get_model(cfg)` |
| Block registry | `src/models/registry.py` → `registry`, `discover_components()` |
| Heads | `src/models/heads.py` → GBM, SDE, NeuralSDE, Horizon, SimpleHorizon, MixtureDensity, VolTermStructure, NeuralBridge |
| Data pipeline | `src/data/market_data_loader.py` → `MarketDataLoader`, `FeatureEngineer`, `DataSource` |
| Trainer | `src/research/trainer.py` → `Trainer`, `DataToModelAdapter` |
| Experiment runner | `src/research/experiment_mgr.py` → `run_experiment(cfg)` |
| Backtest engine | `src/research/backtest.py` → `ChallengerVsChampion` |
| Metrics | `src/research/metrics.py` → `crps_ensemble`, `afcrps_ensemble`, `CRPSMultiIntervalScorer` |
| HF Hub / W&B | `src/tracking/hub_manager.py`, `src/tracking/wandb_logger.py` |
| Configs | `configs/config.yaml`, `configs/model/*.yaml`, `configs/data/*.yaml` |
| Tests | `tests/test_*.py` |

## Architecture Contract

All models follow this pipeline:

```
Input (batch, seq_len, feature_dim)
  → HybridBackbone [composable blocks] → latent (batch, d_model)
  → Head → stochastic parameters
  → Simulation → paths (batch, n_paths, horizon)
```

**Shape invariant**: `SynthModel.forward()` always returns `(paths, mu, sigma)` where `paths.shape == (batch, n_paths, horizon)`.

## Adding New Components

1. **New block**: Create `src/models/components/my_block.py`, decorate with `@registry.register_block("name")`. Auto-discovered.
2. **New head**: Subclass `HeadBase` in `src/models/heads.py`. Add to `HEAD_REGISTRY` and handle in `SynthModel.forward()`.
3. **New data source**: Subclass `DataSource` in `src/data/market_data_loader.py`.
4. **New feature engineer**: Subclass `FeatureEngineer`, implement `feature_dim`, `prepare_cache`, `make_input`, `make_target`, `get_volatility`.

## Key Conventions

- Blocks must preserve `(batch, seq, d_model)` unless `preserves_seq_len=False` is declared
- Volatility floors: always add `1e-6` to sigma outputs
- Log-return clamping: `torch.clamp(log_returns, -MAX_LOG_RETURN_CLAMP, MAX_LOG_RETURN_CLAMP)` where `MAX_LOG_RETURN_CLAMP = 20.0`
- CRPS format: model returns `(batch, n_paths, horizon)`, CRPS expects `(batch, horizon, n_paths)` — use `prepare_paths_for_crps()`
- Type hints required on all public functions; NumPy-style docstrings
- Tests in `tests/` — run with `python -m pytest tests/`

## Running

```bash
# Train
python main.py mode=train training.batch_size=8 training.horizon=12

# Backtest
python main.py mode=backtest data=default_loader backtest.champion_repo_id=username/SN50-Hybrid-Hub

# Tests
python -m pytest tests/ -v
```

## Environment Requirements

- `WANDB_API_KEY` for experiment tracking
- `HF_TOKEN` for Hugging Face Hub uploads
- Python ≥ 3.10, PyTorch ≥ 2.2.0

## ClawHub Skills

This project includes ClawHub agent skills in `.clawhub/`. See `.clawhub/README.md` for the full skill catalog and agent integration guide.
