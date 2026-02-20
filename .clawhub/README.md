# ClawHub Agent Skills — Open Synth Miner

This directory contains structured skill definitions that enable Open Claw agents to interact with the Open Synth Miner framework autonomously and safely.

## Skill Catalog

| Skill | File | Description |
|-------|------|-------------|
| `manifest` | `manifest.yaml` | Project metadata, capabilities, and agent discovery |
| `model-architecture` | `skills/model_architecture.md` | Design, register, and compose hybrid neural architectures |
| `data-pipeline` | `skills/data_pipeline.md` | Data sources, feature engineering, and leak-safe loading |
| `experiment-training` | `skills/experiment_training.md` | Training loops, optimizers, and experiment orchestration |
| `backtest-evaluation` | `skills/backtest_evaluation.md` | Champion-vs-challenger backtesting and CRPS scoring |
| `deployment-publishing` | `skills/deployment_publishing.md` | HF Hub uploads, W&B tracking, model cards, and reporting |
| `guardrails` | `skills/guardrails.md` | Validation rules, safety checks, and anti-patterns |

## How Open Claw Agents Use These Skills

### 1. Discovery

Agents read `manifest.yaml` to understand project capabilities, required tools, and available extension points. The manifest provides structured metadata so agents can determine what actions are possible without reading the full codebase.

### 2. Skill Selection

Each skill file is a self-contained instruction set. Agents select the skill matching their current task:

- **"Add a new block to the model"** → `model_architecture.md`
- **"Load OHLCV data from Hugging Face"** → `data_pipeline.md`
- **"Train a model and log to W&B"** → `experiment_training.md`
- **"Compare my model against the champion"** → `backtest_evaluation.md`
- **"Push model to Hugging Face Hub"** → `deployment_publishing.md`

### 3. Guardrails

Before executing any modification, agents consult `guardrails.md` to validate:
- Shape contracts are preserved
- Numerical stability is maintained
- No data leakage is introduced
- Tests pass after changes

### 4. Composition

Skills can be composed for multi-step workflows:

```
data-pipeline → experiment-training → backtest-evaluation → deployment-publishing
```

## Agent Integration

### For Claude Code / Open Claw Agents

Skills are loaded as context when an agent begins working on this repository. The `CLAUDE.md` file at the project root provides the initial onboarding, and `.clawhub/manifest.yaml` provides structured discovery.

### For Custom Agent Frameworks

Parse `manifest.yaml` for:
- `capabilities`: What the project can do
- `extension_points`: Where agents can add new functionality
- `tools_required`: External dependencies agents need
- `skills`: Detailed instruction files for each capability domain
