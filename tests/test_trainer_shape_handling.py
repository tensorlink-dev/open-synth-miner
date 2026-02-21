"""Tests for Trainer shape handling and path preparation."""
from __future__ import annotations

import pytest
import torch

from src.research.trainer import prepare_paths_for_crps


class TestPreparePathsForCRPS:
    """Test the prepare_paths_for_crps utility function."""

    def test_basic_transpose(self):
        """Should transpose from (batch, n_paths, horizon) to (batch, horizon, n_paths)."""
        paths = torch.randn(4, 50, 60)  # (batch, n_paths, horizon)

        result = prepare_paths_for_crps(paths)

        assert result.shape == (4, 60, 50), f"Expected (4, 60, 50), got {result.shape}"
        # Verify data is correctly transposed
        assert torch.allclose(paths[0, 0, 0], result[0, 0, 0])
        assert torch.allclose(paths[0, 1, 0], result[0, 0, 1])

    def test_single_batch(self):
        """Should work with batch_size=1."""
        paths = torch.randn(1, 100, 30)

        result = prepare_paths_for_crps(paths)

        assert result.shape == (1, 30, 100)

    def test_single_path(self):
        """Should work with n_paths=1."""
        paths = torch.randn(8, 1, 60)

        result = prepare_paths_for_crps(paths)

        assert result.shape == (8, 60, 1)

    def test_preserves_gradients(self):
        """Should preserve gradient flow if paths requires_grad."""
        paths = torch.randn(2, 10, 5, requires_grad=True)

        result = prepare_paths_for_crps(paths)

        assert result.requires_grad, "Should preserve requires_grad"
        # Verify gradients can flow
        loss = result.sum()
        loss.backward()
        assert paths.grad is not None


class TestTrainerShapeContract:
    """Test that Trainer expects SynthModel to return 3D paths."""

    def test_no_2d_handling_needed(self):
        """After refactor, Trainer should NOT need to handle 2D paths.

        This test documents that defensive 2D handling has been removed.
        SynthModel.forward() now enforces 3D output for all head types.
        """
        # If we receive 2D paths, prepare_paths_for_crps will fail
        # (as it should - this indicates a bug in SynthModel)
        paths_2d = torch.randn(4, 60)  # 2D: (batch, horizon)

        with pytest.raises(RuntimeError):
            # transpose expects 3D, will fail on 2D
            result = prepare_paths_for_crps(paths_2d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
