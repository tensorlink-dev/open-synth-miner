"""Tests for AblationExperiment and run_experiment()."""
from __future__ import annotations

from typing import Iterator
from unittest.mock import patch, MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_loader(n_batches: int = 3) -> Iterator:
    """Yield tiny batches that DataToModelAdapter can consume."""
    for _ in range(n_batches):
        yield {
            "inputs": torch.randn(2, 3, 16),
            "target": torch.randn(2, 1, 4),
        }


def _minimal_train_cfg() -> dict:
    """Minimal config dict that AblationExperiment and run_experiment can consume."""
    return {
        "model": {
            "backbone": {
                "_target_": "src.models.factory.HybridBackbone",
                "input_size": 3,
                "d_model": 16,
                "blocks": [
                    {"_target_": "src.models.registry.LSTMBlock", "d_model": 16}
                ],
            },
            "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 16},
        },
        "training": {
            "lr": 1e-3,
            "n_paths": 5,
            "epochs": 1,
            "horizon": 4,
            "batch_size": 2,
            "seq_len": 16,
            "feature_dim": 3,
        },
    }


# ---------------------------------------------------------------------------
# AblationExperiment — train mode
# ---------------------------------------------------------------------------

class TestAblationExperimentTrain:

    def test_run_returns_results_for_each_config(self):
        """run() should produce one result dict per named config."""
        from src.research.ablation import AblationExperiment

        cfg = _minimal_train_cfg()
        exp = AblationExperiment(configs={"model_a": cfg, "model_b": cfg}, mode="train")

        train_loader = list(_fake_loader(3))
        val_loader = list(_fake_loader(2))

        results = exp.run(train_loader=train_loader, val_loader=val_loader)

        assert set(results.keys()) == {"model_a", "model_b"}
        for name, metrics in results.items():
            assert "val_crps" in metrics, f"{name} missing val_crps"
            assert isinstance(metrics["val_crps"], float)
            assert metrics["val_crps"] >= 0

    def test_run_requires_loaders_in_train_mode(self):
        """Missing train/val loaders in train mode should raise ValueError."""
        from src.research.ablation import AblationExperiment

        exp = AblationExperiment(configs={"m": _minimal_train_cfg()}, mode="train")

        with pytest.raises(ValueError, match="train_loader"):
            exp.run()  # no loaders supplied

    def test_results_stored_on_instance(self):
        """After run(), results should be accessible on the experiment instance."""
        from src.research.ablation import AblationExperiment

        exp = AblationExperiment(configs={"only": _minimal_train_cfg()}, mode="train")
        exp.run(train_loader=list(_fake_loader(2)), val_loader=list(_fake_loader(1)))

        assert "only" in exp.results

    def test_invalid_mode_raises(self):
        """Unknown mode should raise ValueError when run() is called."""
        from src.research.ablation import AblationExperiment

        # __init__ does not validate mode; the error surfaces on run().
        exp = AblationExperiment(configs={"m": _minimal_train_cfg()}, mode="foobar")
        with pytest.raises(ValueError, match="Unknown mode"):
            exp.run(train_loader=list(_fake_loader(1)), val_loader=list(_fake_loader(1)))

    def test_device_auto_falls_back_to_cpu(self):
        """device='auto' should resolve to cpu when CUDA is unavailable."""
        from src.research.ablation import AblationExperiment
        import torch

        exp = AblationExperiment(configs={}, mode="train", device="auto")
        if not torch.cuda.is_available():
            assert exp.device.type == "cpu"

    def test_dict_configs_converted_to_dictconfig(self):
        """Plain dicts passed as configs should be wrapped in OmegaConf DictConfig."""
        from src.research.ablation import AblationExperiment
        from omegaconf import DictConfig

        exp = AblationExperiment(configs={"m": _minimal_train_cfg()}, mode="train")
        assert isinstance(exp.configs["m"], DictConfig)


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------

class TestRunExperiment:

    def test_returns_expected_keys(self):
        """run_experiment should return model, metrics, config, run, recipe, block_hash."""
        from src.research.experiment_mgr import run_experiment
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(_minimal_train_cfg())

        with patch("src.research.experiment_mgr.wandb") as mock_wandb, \
             patch("src.tracking.wandb_logger.wandb"):
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.run = MagicMock()
            result = run_experiment(cfg)

        expected_keys = {"model", "metrics", "config", "run", "recipe", "block_hash"}
        assert expected_keys == set(result.keys())

    def test_model_is_synth_model(self):
        """run_experiment should build and return a SynthModel instance."""
        from src.research.experiment_mgr import run_experiment
        from src.models.factory import SynthModel
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(_minimal_train_cfg())

        with patch("src.research.experiment_mgr.wandb"), \
             patch("src.tracking.wandb_logger.wandb"):
            result = run_experiment(cfg)

        assert isinstance(result["model"], SynthModel)

    def test_metrics_has_loss(self):
        """Returned metrics dict should include a 'loss' key."""
        from src.research.experiment_mgr import run_experiment
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(_minimal_train_cfg())

        with patch("src.research.experiment_mgr.wandb"), \
             patch("src.tracking.wandb_logger.wandb"):
            result = run_experiment(cfg)

        assert "loss" in result["metrics"]
        assert isinstance(result["metrics"]["loss"], float)

    def test_block_hash_matches_recipe(self):
        """block_hash should be derived from the backbone blocks recipe."""
        from src.research.experiment_mgr import run_experiment
        from src.models.registry import Registry
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(_minimal_train_cfg())

        with patch("src.research.experiment_mgr.wandb"), \
             patch("src.tracking.wandb_logger.wandb"):
            result = run_experiment(cfg)

        recipe = result["recipe"]
        expected_hash = Registry.recipe_hash(recipe)
        assert result["block_hash"] == expected_hash

    def test_wandb_init_called_once(self):
        """run_experiment should call wandb.init exactly once."""
        from src.research.experiment_mgr import run_experiment
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(_minimal_train_cfg())

        with patch("src.research.experiment_mgr.wandb") as mock_wandb, \
             patch("src.tracking.wandb_logger.wandb"):
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.run = MagicMock()
            run_experiment(cfg)

        mock_wandb.init.assert_called_once()
