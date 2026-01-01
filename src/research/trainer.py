"""Training utilities for Synth miner research runs."""
from __future__ import annotations

from typing import Dict

import torch
import torch.optim as optim

from src.models.factory import SynthModel
from .metrics import crps_ensemble, log_likelihood
from src.tracking.wandb_logger import log_experiment_results


def train_step(
    model: SynthModel,
    batch: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    horizon: int,
    n_paths: int,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad()

    inputs = batch["history"]
    initial_price = batch["initial_price"]
    target = batch["target"].detach()

    paths, mu, sigma = model(inputs, initial_price=initial_price, horizon=horizon, n_paths=n_paths)
    terminal_paths = paths[:, :, -1]
    crps = crps_ensemble(terminal_paths, target)
    loss = crps.mean()
    loss.backward()
    optimizer.step()

    sharpness = terminal_paths.std(dim=1).mean()
    loglik = log_likelihood(terminal_paths, target).mean()
    metrics = {
        "loss": loss.item(),
        "crps": crps.mean().item(),
        "sharpness": sharpness.item(),
        "log_likelihood": loglik.item(),
    }
    return metrics


def evaluate_and_log(
    model: SynthModel,
    batch: Dict[str, torch.Tensor],
    horizon: int,
    n_paths: int,
    step: int,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        inputs = batch["history"]
        initial_price = batch["initial_price"]
        target = batch["target"]
        actual_series = batch.get("actual_series", None)
        paths, mu, sigma = model(inputs, initial_price=initial_price, horizon=horizon, n_paths=n_paths)
        terminal_paths = paths[:, :, -1]
        crps = crps_ensemble(terminal_paths, target)
        sharpness = terminal_paths.std(dim=1).mean()
        loglik = log_likelihood(terminal_paths, target).mean()

        metrics = {
            "loss": crps.mean().item(),
            "crps": crps.mean().item(),
            "sharpness": sharpness.item(),
            "log_likelihood": loglik.item(),
        }
        series = actual_series if actual_series is not None else target.repeat(horizon)
        log_experiment_results(metrics, paths, series, horizon=horizon, step=step)
        return metrics
