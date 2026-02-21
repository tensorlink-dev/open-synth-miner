"""Tests for ChallengerVsChampion backtest engine."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.data.base_dataset import StridedTimeSeriesDataset
from src.models.factory import HybridBackbone, SynthModel
from src.models.heads import GBMHead
from src.models.registry import LSTMBlock
from src.research.backtest import ChallengerVsChampion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synth_model(d_model: int = 16, input_size: int = 1) -> SynthModel:
    bb = HybridBackbone(
        input_size=input_size,
        d_model=d_model,
        blocks=[LSTMBlock(d_model=d_model)],
    )
    return SynthModel(bb, GBMHead(latent_size=d_model))


def _make_data_window(length: int = 20) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.cumsum(torch.randn(length) * 0.01, dim=0) + 100.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChallengerVsChampion:

    def _build_cvc(self, *, model_a: SynthModel, model_b: SynthModel) -> ChallengerVsChampion:
        data_window = _make_data_window(20)
        return ChallengerVsChampion(
            challenger_cfg=model_a,
            champion_cfg=model_b,
            data_window=data_window,
            time_increment=60,
            # ChallengerVsChampion._make_dataloader() hardcodes pred_len=2,
            # so actual_series from the loader is always 2 steps long.
            # horizon must match so the scorer receives equal-length arrays.
            horizon=2,
            n_paths=10,
            device="cpu",
        )

    def test_run_returns_expected_keys(self):
        """run() must return a dict with 'champion', 'challenger', and 'spread' keys."""
        model_a = _make_synth_model()
        model_b = _make_synth_model()

        cvc = self._build_cvc(model_a=model_a, model_b=model_b)

        # Patch get_model so it returns the pre-built models without HF loading,
        # and stub out wandb/log calls.
        with patch("src.research.backtest.get_model", side_effect=[model_b, model_a]), \
             patch("src.research.backtest.wandb") as mock_wandb, \
             patch("src.research.backtest.log_backtest_results"):
            mock_wandb.Table.return_value = MagicMock()
            mock_wandb.Histogram.return_value = MagicMock()
            result = cvc.run(log_to_wandb=True)

        assert set(result.keys()) == {"champion", "challenger", "spread"}

    def test_run_no_wandb_logging(self):
        """run(log_to_wandb=False) must not call wandb at all."""
        model_a = _make_synth_model()
        model_b = _make_synth_model()

        cvc = self._build_cvc(model_a=model_a, model_b=model_b)

        with patch("src.research.backtest.get_model", side_effect=[model_b, model_a]), \
             patch("src.research.backtest.wandb") as mock_wandb, \
             patch("src.research.backtest.log_backtest_results") as mock_log:
            result = cvc.run(log_to_wandb=False)

        mock_wandb.log.assert_not_called()
        mock_log.assert_not_called()
        assert isinstance(result, dict)

    def test_spread_keys_present(self):
        """run() spread dict must contain 'variance_spread' and 'crps_overlap'."""
        model_a = _make_synth_model()
        model_b = _make_synth_model()

        cvc = self._build_cvc(model_a=model_a, model_b=model_b)

        with patch("src.research.backtest.get_model", side_effect=[model_b, model_a]), \
             patch("src.research.backtest.wandb"), \
             patch("src.research.backtest.log_backtest_results"):
            result = cvc.run(log_to_wandb=False)

        assert "variance_spread" in result["spread"]
        assert "crps_overlap" in result["spread"]
        assert result["spread"]["variance_spread"] >= 0.0

    def test_make_dataloader_uses_data_window(self):
        """Internal dataloader should produce batches from the provided data_window."""
        model_a = _make_synth_model()
        cvc = self._build_cvc(model_a=model_a, model_b=model_a)
        dl = cvc._make_dataloader()
        batch = next(iter(dl))
        assert "history" in batch
        assert "actual_series" in batch

    def test_variance_spread_identical_models_is_zero(self):
        """Two identical path tensors should yield zero variance spread."""
        model_a = _make_synth_model()
        cvc = self._build_cvc(model_a=model_a, model_b=model_a)
        # horizon=2 matches _build_cvc config (pred_len=2 hardcoded in _make_dataloader)
        paths = torch.randn(1, 50, 2)
        spread = cvc._variance_spread(paths, paths)
        assert spread == pytest.approx(0.0, abs=1e-6)
