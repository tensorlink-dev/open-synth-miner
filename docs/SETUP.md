# Setup and Running Guide

This guide walks through everything needed to install, configure, and run Open Synth Miner — from a fresh clone to training models, running backtests, and publishing artifacts.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Create a Virtual Environment](#create-a-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Verify the Installation](#verify-the-installation)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Hydra Configuration System](#hydra-configuration-system)
  - [Model Configs](#model-configs)
  - [Data Configs](#data-configs)
- [Running Training](#running-training)
  - [Quick Start with Synthetic Data](#quick-start-with-synthetic-data)
  - [Training with Real Market Data](#training-with-real-market-data)
  - [Overriding Config Parameters](#overriding-config-parameters)
- [Running Backtests](#running-backtests)
- [Running Tests](#running-tests)
- [Using Notebooks](#using-notebooks)
- [Programmatic Usage](#programmatic-usage)
- [GPU Support](#gpu-support)
- [Experiment Tracking](#experiment-tracking)
  - [Weights and Biases](#weights-and-biases)
  - [Hugging Face Hub](#hugging-face-hub)
- [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)

---

## Prerequisites

| Requirement | Minimum Version |
|-------------|-----------------|
| Python      | 3.10+           |
| pip         | 21.0+           |
| Git         | any recent      |

**Hardware:**
- CPU is sufficient for development and small experiments with synthetic data.
- A CUDA-capable GPU is recommended for training on real market data or running large-scale backtests.

**Accounts (optional but recommended):**
- [Weights & Biases](https://wandb.ai/) — experiment tracking, metrics logging, fan charts.
- [Hugging Face](https://huggingface.co/) — model artifact hosting and public dataset access.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/tensorlink-dev/open-synth-miner.git
cd open-synth-miner
```

### Create a Virtual Environment

Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

Or using `uv` (faster alternative):

```bash
uv venv .venv
source .venv/bin/activate
```

### Install Dependencies

**Option A — Editable install (recommended for development):**

```bash
pip install -e .
```

This installs the package in editable mode along with the core dependencies defined in `pyproject.toml`. It also registers the `open-synth-miner` CLI command.

**Option B — Full requirements (includes optional extras):**

```bash
pip install -r requirements.txt
pip install -e .
```

The `requirements.txt` file includes additional packages not in the core dependency list that are used by specific features:

| Package | Purpose |
|---------|---------|
| `torchsde>=0.2.6` | SDE-based heads and evolution blocks |
| `pyarrow>=15.0.0` | Reading Parquet datasets from Hugging Face |
| `pywt>=1.5.0` | Wavelet feature engineering |
| `scikit-learn>=1.3.0` | Regime detection clustering |

If you plan to use SDE heads, real market data, wavelet features, or regime-aware training, install the full requirements.

### Verify the Installation

```bash
# Check that the package is importable
python -c "from open_synth_miner import SynthModel, HybridBackbone; print('OK')"

# Check that the CLI entry point works
open-synth-miner --help

# Inspect the full resolved Hydra config
python main.py --cfg job
```

---

## Configuration

### Environment Variables

Set these before running training or backtest flows that interact with external services:

| Variable | Required | Description |
|----------|----------|-------------|
| `WANDB_API_KEY` | For W&B logging | Your Weights & Biases API key. Get it from [wandb.ai/authorize](https://wandb.ai/authorize). |
| `HF_TOKEN` | For private/gated HF repos | Hugging Face access token. Get it from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). |
| `OHLCV_REPO_ID` | No (has default) | Override the default Hugging Face dataset repo. Defaults to `tensorlink-dev/open-synth-training-data`. |

Set them in your shell:

```bash
export WANDB_API_KEY="your-wandb-key"
export HF_TOKEN="your-hf-token"
```

Or place them in a shell profile (`~/.bashrc`, `~/.zshrc`) for persistence.

Alternatively, you can log in interactively:

```bash
wandb login
huggingface-cli login
```

### Hydra Configuration System

All experiments are driven by [Hydra](https://hydra.cc/). The root config file is `configs/config.yaml`:

```yaml
defaults:
  - model: hybrid_v2       # which model recipe to use
  - data: default_loader    # which data pipeline to use

mode: train                 # "train" or "backtest"
project: synth-miner

training:
  batch_size: 4
  seq_len: 32
  feature_dim: ${data.feature_dim}   # auto-resolved from data config
  horizon: 12
  n_paths: 1000
  lr: 0.001

backtest:
  champion_repo_id: username/SN50-Hybrid-Hub
  time_increment: 60
  horizon: 12
  n_paths: 1000
```

Key Hydra features used:
- **Config groups** (`defaults` list) compose model + data configs.
- **Variable interpolation** (`${data.feature_dim}`) keeps model/data in sync automatically.
- **CLI overrides** let you change any value without editing YAML.
- **`_target_`** entries let Hydra instantiate Python objects directly from config.

### Model Configs

Located in `configs/model/`. Each file defines a full model recipe:

| Config | Architecture | Description |
|--------|-------------|-------------|
| `hybrid_v2.yaml` | TransformerBlock → LSTMBlock → GBMHead | Default. Solid general-purpose setup. |
| `hybrid_with_layernorm.yaml` | Same blocks + auto LayerNorm | Automatically inserts LayerNorm between blocks. |
| `hybrid_manual_layernorm.yaml` | Explicit LayerNorm placement | Manual LayerNormBlock entries in the block list. |

Example — `hybrid_v2.yaml`:

```yaml
model:
  _target_: src.models.factory.SynthModel
  backbone:
    _target_: src.models.factory.HybridBackbone
    input_size: ${training.feature_dim}
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
```

To use a different model config:

```bash
python main.py model=hybrid_with_layernorm
```

### Data Configs

Located in `configs/data/`:

| Config | Source | Features | feature_dim |
|--------|--------|----------|-------------|
| `default_loader.yaml` | `MockDataSource` (synthetic GBM) | Rolling z-scores | 3 |
| `ohlcv_loader.yaml` | `HFOHLCVSource` (Hugging Face) | OHLCV statistics | 16 |

To switch data sources:

```bash
# Synthetic data (no external dependencies)
python main.py data=default_loader

# Real OHLCV market data from Hugging Face
python main.py data=ohlcv_loader
```

When switching to `ohlcv_loader`, the `feature_dim` changes from 3 to 16. The model's `input_size` is automatically updated through Hydra's `${training.feature_dim}` interpolation.

---

## Running Training

### Quick Start with Synthetic Data

The fastest way to run a training loop — no API keys or external data needed:

```bash
python main.py mode=train
```

This uses the default config which:
1. Generates synthetic price data via `MockDataSource`.
2. Builds a TransformerBlock → LSTMBlock → GBMHead model.
3. Trains on synthetic batches and evaluates CRPS / log-likelihood.
4. Attempts to log to W&B and upload to Hugging Face Hub (will skip gracefully if keys are not set).

To reduce resource usage during initial testing:

```bash
python main.py mode=train training.batch_size=2 training.n_paths=128 training.horizon=8
```

### Training with Real Market Data

```bash
export HF_TOKEN="your-hf-token"
export WANDB_API_KEY="your-wandb-key"

python main.py mode=train data=ohlcv_loader training.batch_size=8 training.horizon=12
```

This fetches OHLCV data from the `tensorlink-dev/open-synth-training-data` Hugging Face dataset, engineers 16 features via `OHLCVEngineer`, and trains the model.

### Overriding Config Parameters

Hydra allows overriding any nested config value from the CLI:

```bash
# Change batch size, learning rate, and number of paths
python main.py training.batch_size=16 training.lr=0.0005 training.n_paths=512

# Use a different model config
python main.py model=hybrid_with_layernorm

# Combine model, data, and training overrides
python main.py model=hybrid_v2 data=ohlcv_loader training.batch_size=32 training.horizon=24

# Override deep nested values
python main.py model.head._target_=src.models.heads.SDEHead model.head.latent_size=64
```

To see the full resolved config before running:

```bash
python main.py --cfg job
```

---

## Running Backtests

Backtesting compares a **challenger** model (from your current config) against a **champion** model (loaded from Hugging Face Hub):

```bash
python main.py mode=backtest \
  backtest.champion_repo_id=username/SN50-Hybrid-Hub \
  backtest.horizon=12 \
  backtest.n_paths=1000
```

The backtest engine:
1. Instantiates the challenger from the current model config.
2. Downloads the champion checkpoint from Hugging Face.
3. Evaluates both on the same aligned price window.
4. Computes interval CRPS and variance spread.
5. Logs overlapping fan charts and comparison metrics to W&B.

To run on GPU:

```bash
python main.py mode=backtest backtest.device=cuda:0
```

---

## Running Tests

The test suite lives in `tests/` and uses `pytest`:

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_hybrid_backbone.py -v

# Run a specific test class or method
pytest tests/test_metrics_edge_cases.py::TestAdaptiveIntervals -v

# Run with short traceback for cleaner output
pytest tests/ --tb=short
```

Key test files:

| File | What It Tests |
|------|---------------|
| `test_hybrid_backbone.py` | Shape contracts, LayerNorm insertion, input projection |
| `test_heads_and_simulation.py` | Head outputs, path generation, numerical stability |
| `test_metrics_edge_cases.py` | CRPS scoring, adaptive intervals, edge cases |
| `test_market_data_loader.py` | Data loading, holdout splits, temporal integrity |
| `test_regime_loader.py` | Regime tagging, walk-forward folds, balanced sampling |
| `test_trainer_shape_handling.py` | Training adapter input/output shape contracts |
| `test_block_registry.py` | Component auto-discovery, registry metadata |

---

## Using Notebooks

Research notebooks in `notebooks/` demonstrate end-to-end workflows for specific architectures:

| Notebook | Description |
|----------|-------------|
| `dlinear_train_and_backtest.ipynb` | DLinear block training and backtesting |
| `fedformer_train_and_backtest.ipynb` | FEDformer-based hybrid experiments |
| `fedformer_standalone_train_and_backtest.ipynb` | Standalone FEDformer (non-hybrid) |
| `timesnet_train_and_backtest.ipynb` | TimesNet architecture experiments |
| `timemixer_train_and_backtest.ipynb` | TimeMixer architecture experiments |
| `sde_head_with_sde_block.ipynb` | SDE head with SDE evolution block |
| `walkforward_dlinear_regime.ipynb` | Walk-forward validation with regime awareness |

To run notebooks:

```bash
pip install jupyter
jupyter notebook notebooks/
```

---

## Programmatic Usage

The package can be imported directly for custom workflows:

```python
import torch
from omegaconf import OmegaConf
from open_synth_miner import MarketDataLoader, create_model
from src.data import MockDataSource, ZScoreEngineer

# Define a config programmatically
cfg = OmegaConf.create({
    "model": {
        "_target_": "src.models.factory.SynthModel",
        "backbone": {
            "_target_": "src.models.factory.HybridBackbone",
            "input_size": 3,
            "d_model": 32,
            "validate_shapes": True,
            "blocks": [
                {"_target_": "src.models.registry.TransformerBlock", "d_model": 32, "nhead": 4},
                {"_target_": "src.models.registry.LSTMBlock", "d_model": 32},
            ],
        },
        "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 32},
    },
    "training": {"horizon": 12, "n_paths": 128, "feature_dim": 3},
})

# Create model and data
model = create_model(cfg)
source = MockDataSource(length=512, freq="1h")
engineer = ZScoreEngineer()
loader = MarketDataLoader(
    data_source=source,
    engineer=engineer,
    assets=["BTC"],
    input_len=96,
    pred_len=cfg.training.horizon,
    batch_size=16,
    feature_dim=cfg.training.feature_dim,
)

# Run inference
sample = loader.dataset[0]
history = sample["inputs"].T.unsqueeze(0)
initial_price = torch.ones(history.shape[0])

paths, mu, sigma = model(
    history,
    initial_price=initial_price,
    horizon=cfg.training.horizon,
    n_paths=cfg.training.n_paths,
)
print(paths.shape)  # (batch, n_paths, horizon)
```

---

## GPU Support

PyTorch handles device placement. To use a GPU:

1. Install the CUDA-enabled version of PyTorch (see [pytorch.org/get-started](https://pytorch.org/get-started/locally/)):

   ```bash
   # Example for CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

2. For backtests, pass the device flag:

   ```bash
   python main.py mode=backtest backtest.device=cuda:0
   ```

3. For programmatic usage, move tensors and model to the device:

   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   history = history.to(device)
   initial_price = initial_price.to(device)
   ```

---

## Experiment Tracking

### Weights and Biases

Training and backtest flows automatically log to W&B when `WANDB_API_KEY` is set:

- **Metrics:** CRPS, log-likelihood, variance spread.
- **Tables:** Fan charts comparing simulated paths vs. actuals.
- **Artifacts:** Architecture metadata, config snapshots.
- **Summaries:** Links to Hugging Face model artifacts.

Set up:

```bash
# Option 1: environment variable
export WANDB_API_KEY="your-key"

# Option 2: interactive login
wandb login
```

The project name in W&B is controlled by `project` in `configs/config.yaml` (default: `synth-miner`).

### Hugging Face Hub

After training, `HubManager` uploads:

- `model.pt` — checkpoint.
- `README.md` — auto-generated model card.
- `resolved_config.yaml` — full resolved Hydra config.
- `architecture.json` — architecture metadata.

Artifacts are stored under a taxonomy path: `{backbone_name}/{block_hash}/{run_id}/`.

Set up:

```bash
# Option 1: environment variable
export HF_TOKEN="your-token"

# Option 2: interactive login
huggingface-cli login
```

Update the repo target by setting `backtest.champion_repo_id` in config or via CLI override.

---

## Common Issues and Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

The package must be installed for imports and auto-discovery to work:

```bash
pip install -e .
```

### `TypeError: 'DictConfig' object is not callable`

You are calling the config object like a function. Use `create_model(cfg)` or `hydra.utils.instantiate(cfg.model)` to build the model from config — do not call `cfg.model(...)` directly.

### `401 Unauthorized` when downloading from Hugging Face

The dataset or model repo may be private or gated. Authenticate:

```bash
huggingface-cli login
# or
export HF_TOKEN="your-token"
```

### W&B logging errors or prompts

If you do not want W&B logging, disable it:

```bash
export WANDB_MODE=disabled
```

Or run `wandb offline` to log locally without network calls.

### Shape mismatch errors during training

Ensure `feature_dim` in the data config matches `input_size` in the model config. The default configs handle this automatically via Hydra interpolation (`${training.feature_dim}`), but manual overrides can break the link:

```bash
# This will cause a mismatch — data produces 16 features but model expects 3:
python main.py data=ohlcv_loader model.backbone.input_size=3  # WRONG

# Let interpolation handle it:
python main.py data=ohlcv_loader  # input_size auto-resolves to 16
```

### `torchsde` not found

SDE heads and blocks require `torchsde`, which is in `requirements.txt` but not in the core `pyproject.toml` dependencies:

```bash
pip install torchsde>=0.2.6
```

### Hydra output directories

Hydra creates an `outputs/` directory with timestamped run folders by default. This is normal. Add `outputs/` to your `.gitignore` if it is not already there.

---

## Next Steps

- Read the [Architecture Guide](ARCHITECTURE.md) for design principles and extension patterns.
- See [Hugging Face Market Data Guide](hf_market_data.md) for detailed data loading examples.
- See [RevIN Denormalization Guide](revin_denormalization_guide.md) for normalization details.
- Explore the `notebooks/` directory for end-to-end research workflows.
- Add custom blocks under `src/models/components/` — they are auto-discovered at runtime via the registry.
