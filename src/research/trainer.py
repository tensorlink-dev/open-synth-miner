"""Training utilities and adapters for SynthModel research runs."""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.factory import SynthModel
from src.tracking.wandb_logger import log_experiment_results
from .metrics import CRPSMultiIntervalScorer, crps_ensemble, log_likelihood


def prepare_paths_for_crps(paths: torch.Tensor) -> torch.Tensor:
    """Convert model paths (batch, n_paths, horizon) to CRPS format (batch, horizon, n_paths).

    SynthModel returns paths shaped (batch, n_paths, horizon).
    crps_ensemble expects (batch, horizon, n_paths) with ensemble members in last dimension.

    Parameters
    ----------
    paths : torch.Tensor
        Model output paths of shape (batch, n_paths, horizon)

    Returns
    -------
    torch.Tensor
        Transposed paths of shape (batch, horizon, n_paths) ready for CRPS computation
    """
    return paths.transpose(1, 2)


class DataToModelAdapter:
    """Bridge between leak-safe loader batches and SynthModel inputs."""

    def __init__(self, device: torch.device, *, target_is_log_return: bool = True) -> None:
        self.device = device
        self.target_is_log_return = target_is_log_return

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move tensors to device, transpose, and build price-factor targets.

        - Inputs from :class:`MarketDataLoader` arrive shaped (B, F, T).
        - SynthModel expects (B, T, F), so we transpose.
        - ``target_is_log_return`` toggles whether loader targets are log-returns (default)
          or pre-computed price levels/factors. Log-returns are converted to cumulative
          price factors starting at 1.0.
        - ``initial_price`` is 1.0 so model simulations become relative factors.
        """

        inputs = batch["inputs"].to(self.device)
        target = batch["target"].detach().to(self.device)

        if inputs.ndim != 3:
            raise ValueError(
                f"DataToModelAdapter expects inputs shaped (batch, features, time) "
                f"but got {inputs.ndim}D tensor with shape {tuple(inputs.shape)}."
            )
        if inputs.shape[0] != target.shape[0]:
            raise ValueError(
                f"Batch size mismatch: inputs batch={inputs.shape[0]}, "
                f"target batch={target.shape[0]}."
            )

        history = inputs.transpose(1, 2).contiguous()
        # Squeeze channel dimension: (batch, 1, pred_len) → (batch, pred_len)
        if target.ndim == 3 and target.shape[1] == 1:
            target = target.squeeze(1)
        if self.target_is_log_return:
            target_factors = torch.exp(torch.cumsum(target, dim=-1))
        else:
            target_factors = target
        initial_price = torch.ones(history.shape[0], device=self.device)

        return {
            "history": history,
            "initial_price": initial_price,
            "target_factors": target_factors,
        }


class Trainer:
    """Trainer that adapts leak-safe loader batches for SynthModel."""

    def __init__(
        self,
        model: SynthModel,
        optimizer: optim.Optimizer,
        n_paths: int,
        *,
        device: Optional[torch.device] = None,
        adapter: Optional[DataToModelAdapter] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.n_paths = n_paths
        self.device = device or next(model.parameters()).device
        self.adapter = adapter or DataToModelAdapter(self.device)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step handling shape and semantic adaptation."""

        self.model.train()
        self.optimizer.zero_grad()

        adapted = self.adapter(batch)
        history = adapted["history"]
        initial_price = adapted["initial_price"]
        target = adapted["target_factors"]
        horizon = target.shape[-1]

        paths, mu, sigma = self.model(
            history,
            initial_price=initial_price,
            horizon=horizon,
            n_paths=self.n_paths,
        )
        # SynthModel.forward() enforces (batch, n_paths, horizon) shape for all head types
        sim_paths = prepare_paths_for_crps(paths)
        crps = crps_ensemble(sim_paths, target)
        loss = crps.mean()
        loss.backward()
        self.optimizer.step()

        sharpness = sim_paths.std(dim=-1).mean()
        loglik = log_likelihood(sim_paths, target).mean()
        return {
            "loss": loss.item(),
            "crps": crps.mean().item(),
            "sharpness": sharpness.item(),
            "log_likelihood": loglik.item(),
            "mu": mu.mean().item(),
            "sigma": sigma.mean().item(),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run CRPS validation over a holdout loader."""

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in dataloader:
            adapted = self.adapter(batch)
            history = adapted["history"]
            initial_price = adapted["initial_price"]
            target = adapted["target_factors"]
            horizon = target.shape[-1]

            paths, _, _ = self.model(
                history,
                initial_price=initial_price,
                horizon=horizon,
                n_paths=self.n_paths,
            )
            # SynthModel.forward() enforces (batch, n_paths, horizon) shape for all head types
            sim_paths = prepare_paths_for_crps(paths)
            crps = crps_ensemble(sim_paths, target)

            total_loss += crps.mean().item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        return {"val_crps": avg_loss}


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
    time_increment: int = 60,
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
        series = actual_series if actual_series is not None else target.unsqueeze(-1).repeat(1, horizon)

        if actual_series is not None:
            scorer = CRPSMultiIntervalScorer(
                time_increment=time_increment,
                adaptive=True,  # Automatically adapt intervals to horizon length
            )
            interval_totals = []
            for batch_idx in range(paths.shape[0]):
                total, _ = scorer(paths[batch_idx], actual_series[batch_idx])
                interval_totals.append(total)
            metrics["multi_interval_crps"] = float(np.mean(interval_totals))

        log_experiment_results(metrics, paths, series, horizon=horizon, step=step)
        return metrics
