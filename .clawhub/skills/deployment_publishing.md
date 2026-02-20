# Skill: Deployment & Publishing

## Purpose
Upload trained models to Hugging Face Hub, log experiments to Weights & Biases, generate model cards, and produce shareable reports.

## When to Use
- User wants to push a trained model to Hugging Face Hub
- User needs to set up W&B tracking for experiments
- User asks about model cards, artifacts, or shareable reports
- User wants to understand the publish pipeline

---

## Core Concepts

### Publishing Pipeline

```
Trained Model
    │
    ▼
HubManager(run, backbone_name, head_name, block_hash, recipe)
    │
    ├── save_and_push(model, crps_score)
    │       ├── Save model.pt (state dict)
    │       ├── Write README.md (model card)
    │       ├── Write resolved_config.yaml
    │       ├── Log architecture.json as W&B artifact
    │       ├── Upload folder to HF Hub
    │       └── Update W&B summary with HF link
    │
    └── get_shareable_report(crps_score, hf_link)
            └── Formatted text for sharing (X/Twitter, etc.)
```

### Key Classes

| Class | Location | Role |
|-------|----------|------|
| `HubManager` | `src/tracking/hub_manager.py` | HF Hub uploads + W&B bridge |
| `log_experiment_results` | `src/tracking/wandb_logger.py` | W&B metrics + fan chart logging |
| `log_backtest_results` | `src/tracking/wandb_logger.py` | W&B backtest result logging |

---

## Task: Publish a Model After Training

The standard flow via `_train_flow()` in `main.py` handles publishing automatically:

```bash
python main.py mode=train
# → Trains model
# → Logs metrics to W&B
# → Uploads to HF Hub at username/SN50-Hybrid-Hub/<backbone>/<hash>/<run_id>/
# → Prints shareable report
```

### Manual Publishing

```python
import wandb
from src.tracking.hub_manager import HubManager
from omegaconf import OmegaConf

# After training...
cfg = OmegaConf.load("configs/config.yaml")
resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

manager = HubManager(
    run=wandb.run,                           # Active W&B run
    backbone_name="HybridBackbone",          # Backbone identifier
    head_name="GBMHead",                     # Head identifier
    block_hash="abc123def456",               # Registry recipe hash
    recipe=[                                  # Block list for metadata
        {"_target_": "src.models.registry.TransformerBlock", "d_model": 32},
        {"_target_": "src.models.registry.LSTMBlock", "d_model": 32},
    ],
    architecture_graph=resolved_cfg.get("model", {}).get("backbone", {}),
    resolved_config=resolved_cfg,
    repo_id="username/SN50-Hybrid-Hub",      # HF repo
)

# Upload
hf_link = manager.save_and_push(model=trained_model, crps_score=0.042)
print(f"Published to: {hf_link}")

# Generate shareable report
report = manager.get_shareable_report(crps_score=0.042, hf_link=hf_link)
print(report)
```

### Shareable Report Format

```
Model: HybridBackbone + GBMHead
Hybrid Recipe: [{"_target_": "...TransformerBlock", ...}, ...]
CRPS Score: 0.042000
HF Folder: https://huggingface.co/username/SN50-Hybrid-Hub/tree/main/HybridBackbone/abc123def456/run_id
W&B Dashboard: https://wandb.ai/...
```

---

## Task: Load a Published Model

```python
from src.models.factory import get_model
from src.models.registry import discover_components

discover_components("src/models/components")

# Load from HF Hub
cfg = {
    "model": {
        "hf_repo_id": "username/SN50-Hybrid-Hub",
        "architecture": {
            "_target_": "src.models.factory.SynthModel",
            "backbone": {...},  # Must match the published architecture
            "head": {...},
        },
    }
}

model = get_model(cfg)
# get_model downloads the repo, finds model.pt, loads state dict
```

---

## Task: Set Up W&B Tracking

### Environment Setup

```bash
# Set API key
export WANDB_API_KEY=your_key_here

# Or login interactively
wandb login
```

### What Gets Logged

The `run_experiment()` flow logs:

| Metric | Description |
|--------|-------------|
| `loss` | CRPS loss value |
| `crps` | CRPS metric |
| `sharpness` | Ensemble standard deviation |
| `log_likelihood` | Gaussian log-likelihood |
| `multi_interval_crps` | Sum across all scoring intervals |

### Fan Charts

`log_experiment_results()` creates W&B tables with fan-chart data:
- Simulated path percentiles (5th, 25th, 50th, 75th, 95th)
- Actual price series overlay
- Logged as `wandb.Table` for interactive visualization

### Architecture Artifacts

Each run logs:
- `architecture.json`: Full block recipe and configuration
- W&B Artifact type: `"architecture"`
- Linked to the W&B run for lineage tracking

---

## Task: Configure W&B Project Settings

```yaml
# configs/config.yaml
project: synth-miner      # W&B project name
```

The W&B run group is auto-generated:
```
group=backbone=<backbone_target>_head=<head_target>_recipe=<hash>
```

---

## HF Hub Directory Structure

Published models follow this taxonomy:

```
username/SN50-Hybrid-Hub/
└── <backbone_name>/
    └── <block_hash>/
        └── <wandb_run_id>/
            ├── model.pt              # State dict
            ├── README.md             # Auto-generated model card
            ├── resolved_config.yaml  # Full Hydra config
            └── architecture.json     # Block recipe
```

### Model Card Contents

Auto-generated from `HubManager._write_model_card()`:
- Tags: `bittensor`, `sn50`, `time-series`
- Metadata: backbone, head, CRPS score, hybrid recipe
- Links: W&B run URL, HF folder URL
- Full recipe in JSON format

---

## Environment Requirements

| Variable | Required For | How to Set |
|----------|-------------|------------|
| `WANDB_API_KEY` | W&B logging | `export WANDB_API_KEY=...` or `wandb login` |
| `HF_TOKEN` | HF Hub uploads | `export HF_TOKEN=...` or `huggingface-cli login` |

### Offline Mode

If W&B is unavailable:
```bash
export WANDB_MODE=offline
python main.py mode=train
```
Metrics are logged locally and can be synced later with `wandb sync`.

---

## Common Pitfalls

1. **Missing HF_TOKEN**: Upload fails with 401. Run `huggingface-cli login` first.
2. **Wrong repo_id**: Default is `username/SN50-Hybrid-Hub`. Change it if publishing to a different repo.
3. **Architecture mismatch on load**: When loading from HF, the `architecture` config must match what was used during training. Keys, block order, and dimensions must be identical.
4. **W&B not initialized**: `run_experiment()` calls `wandb.init()`. If running manually, ensure you call `wandb.init()` before `HubManager`.
5. **Large model files**: HF Hub has file size limits. For very large models, consider using `safetensors` format.
