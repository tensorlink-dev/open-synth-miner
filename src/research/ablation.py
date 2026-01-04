"""Unified experiment class for training and backtesting ablations."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.models.factory import create_model, get_model
from src.research.backtest_runner import BacktestRunner
from src.research.metrics import CRPSMultiIntervalScorer
from src.research.trainer import Trainer


class AblationExperiment:
    """
    Single experiment unit that can either TRAIN a new model or BACKTEST an existing one.

    Modes:
      - "train": Instantiates a fresh model from config, trains it, and returns val metrics.
      - "backtest": Loads a model (local or HF), runs backtest on holdout data, returns metrics.
    """

    def __init__(
        self,
        name: str,
        cfg: Union[DictConfig, Dict[str, Any]],
        mode: str = "train",
        device: str = "auto",
    ) -> None:
        self.name = name
        self.cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
        self.mode = mode

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[torch.nn.Module] = None
        self.metrics: Dict[str, Any] = {}

    def setup_model(self) -> torch.nn.Module:
        """Initialize the model based on mode."""
        print(f"[{self.name}] Initializing model on {self.device}...")

        if self.mode == "train":
            # Create fresh model for training
            self.model = create_model(self.cfg).to(self.device)
        elif self.mode == "backtest":
            # Load existing model (supports "hf_repo_id" in config for HF loading)
            self.model = get_model(self.cfg).to(self.device)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return self.model

    def run(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Execute the experiment flow."""
        self.setup_model()

        if self.mode == "train":
            if not train_loader or not val_loader:
                raise ValueError("Train mode requires train_loader and val_loader")
            return self._run_training(train_loader, val_loader)

        if self.mode == "backtest":
            loader = test_loader or val_loader
            if not loader:
                raise ValueError("Backtest mode requires test_loader (or val_loader)")
            return self._run_backtest(loader)

        raise ValueError(f"Unknown mode: {self.mode}")

    def _run_training(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Train loop using src.research.trainer.Trainer."""
        lr = self.cfg.training.get("lr", 1e-3)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            n_paths=self.cfg.training.get("n_paths", 100),
            device=self.device,
        )

        epochs = self.cfg.training.get("epochs", 5)
        best_crps = float("inf")

        for epoch in range(epochs):
            train_loss = 0.0
            batches = 0

            for batch in train_loader:
                step_metrics = trainer.train_step(batch)
                train_loss += step_metrics["loss"]
                batches += 1

            avg_train = train_loss / max(batches, 1)
            val_metrics = trainer.validate(val_loader)

            print(
                f"[{self.name}] Epoch {epoch + 1} | Train: {avg_train:.4f} | Val CRPS: {val_metrics['val_crps']:.4f}"
            )

            if val_metrics["val_crps"] < best_crps:
                best_crps = val_metrics["val_crps"]
                self.metrics = val_metrics

        return self.metrics

    def _run_backtest(self, loader: DataLoader) -> Dict[str, Any]:
        """Backtest loop using src.research.backtest_runner.BacktestRunner."""
        print(f"[{self.name}] Starting backtest...")

        scorer = CRPSMultiIntervalScorer(
            time_increment=self.cfg.backtest.get("time_increment", 60)
        )

        runner = BacktestRunner(
            models={self.name: self.model},
            dataloader=loader,
            scorer=scorer,
            device=self.device,
        )

        results = runner.run(
            horizon=self.cfg.training.get("horizon", 24),
            n_paths=self.cfg.training.get("n_paths", 100),
        )

        self.metrics = results[self.name]
        print(f"[{self.name}] Results: {self.metrics}")
        return self.metrics
