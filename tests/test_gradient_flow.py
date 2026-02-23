"""Gradient-flow tests for backbone blocks and the Trainer training loop.

These verify that:
- Backward passes propagate gradients through every parameter of each block.
- Calling train_step() actually changes model weights.
- Running multiple training steps on a tiny dataset reduces loss.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from osa.models.factory import HybridBackbone, SynthModel
from osa.models.heads import GBMHead, SDEHead
from osa.models.registry import (
    LSTMBlock,
    SDEEvolutionBlock,
    TransformerBlock,
)
from osa.research.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _randn(batch: int = 2, seq: int = 16, d: int = 32) -> torch.Tensor:
    return torch.randn(batch, seq, d)


def _assert_gradients_populated(module: nn.Module, context: str) -> None:
    """Assert every parameter in *module* has a non-None gradient."""
    for name, param in module.named_parameters():
        assert param.grad is not None, (
            f"[{context}] No gradient for parameter '{name}'"
        )
        assert torch.isfinite(param.grad).all(), (
            f"[{context}] Non-finite gradient for parameter '{name}'"
        )


# ---------------------------------------------------------------------------
# Core backbone blocks
# ---------------------------------------------------------------------------

class TestTransformerBlockGradients:

    def test_gradient_flows_to_all_parameters(self):
        block = TransformerBlock(d_model=32, nhead=4, dropout=0.0)
        x = _randn()
        out = block(x)
        out.sum().backward()
        _assert_gradients_populated(block, "TransformerBlock")

    def test_gradient_flows_from_input(self):
        block = TransformerBlock(d_model=32, nhead=4, dropout=0.0)
        x = torch.randn(2, 16, 32, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_output_depends_on_input(self):
        """Output should differ for different inputs (block is not constant)."""
        block = TransformerBlock(d_model=32, nhead=4, dropout=0.0)
        block.eval()
        x1 = _randn()
        x2 = _randn()
        assert not torch.allclose(block(x1), block(x2)), (
            "TransformerBlock should produce different outputs for different inputs"
        )


class TestLSTMBlockGradients:

    def test_gradient_flows_to_all_parameters(self):
        block = LSTMBlock(d_model=32, num_layers=2)
        x = _randn()
        out = block(x)
        out.sum().backward()
        _assert_gradients_populated(block, "LSTMBlock")

    def test_output_is_sequence_dependent(self):
        """LSTM output at step T should differ when earlier timesteps differ."""
        block = LSTMBlock(d_model=32)
        block.eval()
        x1 = _randn()
        x2 = x1.clone()
        x2[:, 0, :] += 5.0  # perturb only the first timestep
        # All output steps should differ because LSTM carries state forward.
        assert not torch.allclose(block(x1), block(x2)), (
            "LSTMBlock output should depend on full input history"
        )


class TestSDEEvolutionBlockGradients:

    def test_gradient_flows_to_all_parameters(self):
        block = SDEEvolutionBlock(d_model=32, hidden=64, dropout=0.0)
        x = _randn()
        out = block(x)
        out.sum().backward()
        _assert_gradients_populated(block, "SDEEvolutionBlock")

    def test_residual_connection(self):
        """Block uses x + net(x); zeroing weights should make output = input."""
        block = SDEEvolutionBlock(d_model=32, hidden=64, dropout=0.0)
        with torch.no_grad():
            for p in block.net.parameters():
                p.zero_()
        x = _randn()
        out = block(x)
        assert torch.allclose(out, x, atol=1e-6), (
            "With zeroed weights the residual block should be an identity"
        )


# ---------------------------------------------------------------------------
# HybridBackbone gradient flow (end-to-end)
# ---------------------------------------------------------------------------

class TestHybridBackboneGradients:

    def test_full_pipeline_gradient_flow(self):
        """Gradients must reach all parameters through a multi-block backbone + head."""
        bb = HybridBackbone(
            input_size=3,
            d_model=32,
            blocks=[
                TransformerBlock(d_model=32, nhead=4, dropout=0.0),
                LSTMBlock(d_model=32),
                SDEEvolutionBlock(d_model=32, dropout=0.0),
            ],
        )
        head = GBMHead(latent_size=32)
        model = SynthModel(bb, head)

        x = torch.randn(2, 16, 3)
        price = torch.ones(2) * 100.0

        with torch.enable_grad():
            paths, mu, sigma = model(x, price, horizon=4, n_paths=5)

        loss = paths.sum()
        loss.backward()
        _assert_gradients_populated(model, "HybridBackbone+GBMHead")


# ---------------------------------------------------------------------------
# Trainer: weights change and loss decreases
# ---------------------------------------------------------------------------

class TestTrainerTrainingLoop:

    def _make_trainer(self, n_paths: int = 10) -> tuple[Trainer, SynthModel]:
        bb = HybridBackbone(
            input_size=3,
            d_model=16,
            blocks=[LSTMBlock(d_model=16)],
        )
        model = SynthModel(bb, GBMHead(latent_size=16))
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        trainer = Trainer(model, opt, n_paths=n_paths)
        return trainer, model

    def _batch(self) -> dict:
        return {
            "inputs": torch.randn(4, 3, 16),
            "target": torch.randn(4, 1, 4),
        }

    def test_train_step_changes_weights(self):
        """A single train_step should update at least one model parameter."""
        trainer, model = self._make_trainer()
        params_before = {n: p.clone() for n, p in model.named_parameters()}

        trainer.train_step(self._batch())

        changed = [
            n for n, p in model.named_parameters()
            if not torch.allclose(p, params_before[n])
        ]
        assert len(changed) > 0, (
            "No parameters changed after train_step — optimizer may not be working"
        )

    def test_train_step_gradients_populated(self):
        """After train_step, every parameter should have received a gradient."""
        trainer, model = self._make_trainer()
        trainer.train_step(self._batch())
        _assert_gradients_populated(model, "Trainer.train_step")

    def test_loss_decreases_over_multiple_steps(self):
        """Running 20 train_steps on the same batch should reduce loss."""
        torch.manual_seed(0)
        trainer, _ = self._make_trainer(n_paths=20)
        batch = self._batch()

        first_loss = trainer.train_step(batch)["loss"]
        for _ in range(19):
            last_metrics = trainer.train_step(batch)
        last_loss = last_metrics["loss"]

        assert last_loss < first_loss, (
            f"Loss did not decrease after 20 steps: {first_loss:.4f} -> {last_loss:.4f}"
        )

    def test_validate_returns_finite_crps(self):
        """validate() must return a finite, non-negative val_crps."""
        trainer, _ = self._make_trainer()

        def _loader():
            for _ in range(3):
                yield self._batch()

        metrics = trainer.validate(_loader())
        val_crps = metrics["val_crps"]
        assert isinstance(val_crps, float)
        assert torch.isfinite(torch.tensor(val_crps)), "val_crps should be finite"
        assert val_crps >= 0.0
        # A well-implemented model with n_paths=10 producing GBM paths will have
        # CRPS well below 1e6; this catches a broken implementation that returns huge values.
        assert val_crps < 1e6, f"val_crps suspiciously large: {val_crps}"

    def test_trainer_crps_alpha_none_uses_standard_crps(self):
        """crps_alpha=None should use standard CRPS (not afCRPS) without errors."""
        bb = HybridBackbone(
            input_size=3, d_model=16, blocks=[LSTMBlock(d_model=16)]
        )
        model = SynthModel(bb, SDEHead(latent_size=16))
        opt = torch.optim.Adam(model.parameters())
        trainer = Trainer(model, opt, n_paths=5, crps_alpha=None)
        metrics = trainer.train_step(self._batch())
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] >= 0
