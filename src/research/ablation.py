"""Unified experiment class for training and backtesting multiple ablation configurations."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from osa.models.factory import create_model, get_model
from osa.research.backtest_runner import BacktestRunner
from osa.research.metrics import CRPSMultiIntervalScorer
from osa.research.trainer import Trainer


class AblationExperiment:
    """
    Experiment suite that runs multiple model configurations sequentially.

    Modes:
      - "train": Instantiates fresh models from configs, trains them, and returns val metrics.
      - "backtest": Loads existing models (local or HF), runs backtests, and returns metrics.
    """

    def __init__(
        self,
        configs: Dict[str, Union[DictConfig, Dict[str, Any]]],
        mode: str = "train",
        device: str = "auto",
    ) -> None:
        """
        Args:
            configs: Dictionary mapping experiment names to their Hydra/OmegaConf configurations.
            mode: "train" or "backtest".
            device: "cpu", "cuda", or "auto".
        """
        self.configs = {
            name: (cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg))
            for name, cfg in configs.items()
        }
        self.mode = mode

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.results: Dict[str, Dict[str, float]] = {}

    def _setup_model(self, name: str, cfg: DictConfig) -> torch.nn.Module:
        """Initialize a single model based on mode and config."""
        print(f"[{name}] Initializing model on {self.device}...")

        if self.mode == "train":
            # Create fresh model for training
            model = create_model(cfg).to(self.device)
        elif self.mode == "backtest":
            # Load existing model (supports "hf_repo_id" in config for HF loading)
            model = get_model(cfg).to(self.device)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return model

    def run(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Execute the experiment flow for all configured variants.
        
        Returns:
            Dict[str, Dict[str, float]]: A dictionary mapping experiment names to their metric results.
        """
        self.results = {}

        for name, cfg in self.configs.items():
            model = self._setup_model(name, cfg)
            
            if self.mode == "train":
                if not train_loader or not val_loader:
                    raise ValueError("Train mode requires train_loader and val_loader")
                metrics = self._run_training(name, model, cfg, train_loader, val_loader)
            elif self.mode == "backtest":
                loader = test_loader or val_loader
                if not loader:
                    raise ValueError("Backtest mode requires test_loader (or val_loader)")
                metrics = self._run_backtest(name, model, cfg, loader)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            self.results[name] = metrics
            print(f"[{name}] Completed. Metrics: {metrics}\n")

        return self.results

    def _run_training(
        self, 
        name: str, 
        model: torch.nn.Module, 
        cfg: DictConfig, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Train a single model variant."""
        lr = cfg.training.get("lr", 1e-3)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            n_paths=cfg.training.get("n_paths", 100),
            device=self.device,
        )

        epochs = cfg.training.get("epochs", 5)
        best_crps = float("inf")
        best_metrics: Dict[str, float] = {}

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
                f"[{name}] Epoch {epoch + 1} | Train: {avg_train:.4f} | Val CRPS: {val_metrics['val_crps']:.4f}"
            )

            if val_metrics["val_crps"] < best_crps:
                best_crps = val_metrics["val_crps"]
                best_metrics = val_metrics

        return best_metrics

    def _run_backtest(
        self, 
        name: str, 
        model: torch.nn.Module, 
        cfg: DictConfig, 
        loader: DataLoader
    ) -> Dict[str, float]:
        """Backtest a single model variant."""
        print(f"[{name}] Starting backtest...")

        scorer = CRPSMultiIntervalScorer(
            time_increment=cfg.backtest.get("time_increment", 60),
            adaptive=True,  # Automatically adapt intervals to horizon length
        )

        runner = BacktestRunner(
            models={name: model},
            dataloader=loader,
            scorer=scorer,
            device=self.device,
        )

        results = runner.run(
            horizon=cfg.training.get("horizon", 24),
            n_paths=cfg.training.get("n_paths", 100),
        )

        # BacktestRunner returns Dict[model_name, Dict[interval, score]]
        # We unwrap it to return just the metrics for this model
        return results[name]
