# Open Synth Miner

Open Synth Miner is a Hydra-driven research framework for Bittensor SN50 (Synth) that focuses on hybrid neural architectures, stochastic path simulation, and "build in public" tracking. Models are assembled from registry-driven blocks, trained to generate 1,000 differentiable price paths, and logged to Weights & Biases while automatically publishing artifacts to the Hugging Face Hub.

## Key Features
- **Compositional Registry & Auto-Discovery**: Register components, blocks, and hybrids with decorators and load them recursively so recipes stay declarative even with thousands of modules. Hybrid backbones stitch blocks from YAML blueprints rather than monolithic classes, and legacy LSTM/Transformer setups translate into block recipes seamlessly.
- **Hydra-Native Model Recipes**: Define architectures via `_target_`-based configs (e.g., `configs/model/hybrid_v2.yaml`) that can include parallel fusion, SDE blocks, or other experimental modules. The universal factory supports fresh instantiation or loading from Hugging Face when `hf_repo_id` is provided.
- **Data Loading for Research & Backtesting**: The `MarketDataLoader` (Hydra-instantiable) produces windowed price tensors plus optional covariates for training or champion-vs-challenger comparisons, with deterministic slices for reproducible backtests.
- **Training Pipeline with Differentiable Simulation**: `run_experiment` builds models, simulates 1,000 paths end-to-end, logs CRPS/log-likelihood metrics, and surfaces architecture metadata for downstream publishing.
- **Challenger vs. Champion Backtests**: The backtest engine instantiates a challenger model and a champion loaded from the Hugging Face Hub, evaluates both on aligned datasets, computes interval CRPS/variance spread, and logs overlapping fan charts to W&B.
- **Hub & W&B Automation**: `HubManager` saves checkpoints under taxonomy-rich folders, uploads to `username/SN50-Hybrid-Hub`, generates model cards that link back to W&B runs, updates W&B summaries with HF links, and prints shareable reports for X/Twitter.

## Configuration
Hydra drives all experiments. The root config selects model/data defaults and toggles train vs. backtest modes:
```yaml
# configs/config.yaml
defaults:
  - model: hybrid_v2
  - data: default_loader

mode: train
project: synth-miner

training:
  batch_size: 4
  seq_len: 32
  feature_dim: 4
  horizon: 12
  n_paths: 1000
  lr: 0.001

backtest:
  champion_repo_id: username/SN50-Hybrid-Hub
  time_increment: 60
  horizon: 12
  n_paths: 1000
```

Model recipes use `_target_` entries so Hydra can instantiate nested blocks automatically:
```yaml
# configs/model/hybrid_v2.yaml
model:
  _target_: src.models.factory.SynthModel
  backbone:
    _target_: src.models.factory.HybridBackbone
    input_size: 4
    d_model: 32
    blocks:
      - _target_: src.models.registry.TransformerBlock
        d_model: 32
        nhead: 4
      - _target_: src.models.registry.LSTMBlock
        d_model: 32
        num_layers: 1
  head:
    _target_: src.models.heads.GBMHead
    latent_size: 32
    sigma_min: 0.01
    sigma_max: 0.5
```

Data presets remain Hydra-friendly; `configs/data/default_loader.yaml` instantiates a lightweight synthetic loader with symbol/timeframe/window settings that can be swapped for real data sources.

## Usage
1. **Discover components automatically and run training**
   ```bash
   python main.py mode=train
   ```
   The entrypoint discovers registry components, builds the model from the Hydra config, runs training/evaluation, uploads artifacts to Hugging Face via `HubManager`, updates the W&B summary with the HF link, and prints a shareable X/Twitter blurb.

2. **Champion vs. Challenger backtest**
   ```bash
   python main.py mode=backtest backtest.champion_repo_id=username/SN50-Hybrid-Hub
   ```
   This loads a champion from the configured HF repo, instantiates the challenger from the current config, runs both on the same backtest window, logs overlapping fans/metrics to W&B, and prints the aggregated results.

3. **Customize architectures**
   - Add new blocks/components under `src/models/components/`; the auto-discovery step will register them.
   - Update `configs/model/*.yaml` to declare new hybrid recipes (including parallel fusion or partial blocks) without touching code.

4. **Programmatic usage (import as a package)**
   ```python
   from omegaconf import OmegaConf
   from open_synth_miner import create_model, MarketDataLoader

   cfg = OmegaConf.create(
       {
           "model": {
               "backbone": {
                   "_target_": "src.models.factory.HybridBackbone",
                   "input_size": 4,
                   "d_model": 32,
                   "blocks": [
                       {"_target_": "src.models.registry.TransformerBlock", "d_model": 32, "nhead": 4},
                       {"_target_": "src.models.registry.LSTMBlock", "d_model": 32},
                   ],
               },
               "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 32},
           },
           "training": {"horizon": 12, "n_paths": 128},
       }
   )

   # Build the model and prepare a quick window of synthetic prices.
   model = create_model(cfg)
   loader = MarketDataLoader(symbols=["BTC", "ETH", "SOL", "ATOM"], timeframe="1h", window_size=64)
   window = loader.latest_window()
   history = window["prices"].unsqueeze(0)  # [batch, seq_len, feature_dim]
   initial_price = history[:, -1, 0]

   paths, mu, sigma = model(history, initial_price=initial_price, horizon=cfg.training.horizon, n_paths=cfg.training.n_paths)
   print(paths.shape)  # (batch, n_paths, horizon)
   ```
   The top-level package now exposes factories, registries, and data utilities so you can prototype without drilling into submodules.

## Directory Highlights
- `src/models/registry.py`: Component/block/hybrid registries plus recursive discovery for decorator-based registration.
- `src/models/factory.py`: Hydra-driven model creation (fresh or HF-loaded) and hybrid backbone wiring from recipes.
- `src/data/loader.py`: Hydra-instantiable market data loader with reproducible slicing for training/backtests.
- `src/research/backtest.py`: Challenger-vs-champion engine computing interval CRPS and variance spread with W&B logging.
- `src/tracking/hub_manager.py`: Hugging Face + W&B bridge that uploads taxonomy-structured artifacts and emits shareable reports.
- `configs/`: Hydra configs for defaults, data presets, and hybrid model recipes.

## Notes
- Experiments rely on W&B for metrics, tables (fan charts), and artifact tracking; ensure `WANDB_API_KEY` is set.
- Hugging Face uploads use `huggingface_hub.HfApi`; set `HF_TOKEN` (or login) before running training flows that push artifacts.
