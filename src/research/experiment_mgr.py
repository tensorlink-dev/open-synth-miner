"""Configuration-driven experiment manager."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.optim as optim
import wandb
from omegaconf import OmegaConf

from src.models.factory import create_model
from src.models.registry import registry
from src.research.trainer import evaluate_and_log, train_step


def load_config(path: str):
    return OmegaConf.load(path)


def _build_dummy_batch(training_cfg: dict) -> dict:
    batch_size = training_cfg.get("batch_size", 1)
    seq_len = training_cfg.get("seq_len", 20)
    feature_dim = training_cfg.get("feature_dim", 4)
    history = torch.randn(batch_size, seq_len, feature_dim)
    initial_price = torch.full((batch_size,), training_cfg.get("initial_price", 100.0))
    target = initial_price + torch.randn(batch_size) * training_cfg.get("target_std", 1.0)
    horizon = training_cfg.get("horizon", 10)
    actual_series = initial_price.unsqueeze(-1) * torch.exp(
        torch.linspace(0, 0.01 * horizon, steps=horizon)
    )
    return {
        "history": history,
        "initial_price": initial_price,
        "target": target,
        "actual_series": actual_series,
    }


def run_experiment(config: Any) -> Dict[str, Any]:
    cfg = config if not isinstance(config, str) else load_config(config)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    recipe = resolved_cfg.get("model", {}).get("backbone", {}).get("blocks", [])
    block_hash = registry.recipe_hash(recipe) if recipe else "na"

    model = create_model(cfg)
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.get("lr", 1e-3))

    wandb.init(
        project=cfg.get("project", "synth-miner"),
        config=resolved_cfg,
        group=f"backbone={model_cfg.get('backbone', {}).get('_target_', 'unknown')}_head={model_cfg.get('head', {}).get('_target_', 'unknown')}_recipe={block_hash}",
    )

    batch = _build_dummy_batch(training_cfg)

    train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        horizon=training_cfg.get("horizon", 10),
        n_paths=training_cfg.get("n_paths", 1000),
    )
    metrics = evaluate_and_log(
        model=model,
        batch=batch,
        horizon=training_cfg.get("horizon", 10),
        n_paths=training_cfg.get("n_paths", 1000),
        step=0,
    )
    return {
        "model": model,
        "metrics": metrics,
        "config": resolved_cfg,
        "run": wandb.run,
        "recipe": recipe,
        "block_hash": block_hash,
    }
