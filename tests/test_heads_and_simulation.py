"""Tests for head modules and simulation functions."""
from __future__ import annotations

import pytest
import torch

from src.models.factory import (
    MAX_LOG_RETURN_CLAMP,
    simulate_bridge_paths,
    simulate_gbm_paths,
    simulate_horizon_paths,
    simulate_t_horizon_paths,
)
from src.models.heads import (
    GBMHead, HorizonHead, SimpleHorizonHead, CLTHorizonHead,
    StudentTHorizonHead, NeuralBridgeHead, SDEHead,
)


class TestHeadOutputShapes:
    """Test that all heads return expected shapes."""

    def test_gbm_head_output_shape(self):
        """GBMHead should return (mu, sigma) both shaped (batch,)."""
        head = GBMHead(latent_size=128)
        h_t = torch.randn(4, 128)
        mu, sigma = head(h_t)
        assert mu.shape == (4,), f"Expected (4,), got {mu.shape}"
        assert sigma.shape == (4,), f"Expected (4,), got {sigma.shape}"
        assert (sigma > 0).all(), "Sigma should be positive"

    def test_sde_head_output_shape(self):
        """SDEHead should return (mu, sigma) both shaped (batch,)."""
        head = SDEHead(latent_size=128, hidden=64)
        h_t = torch.randn(4, 128)
        mu, sigma = head(h_t)
        assert mu.shape == (4,), f"Expected (4,), got {mu.shape}"
        assert sigma.shape == (4,), f"Expected (4,), got {sigma.shape}"
        assert (sigma > 0).all(), "Sigma should be positive"

    def test_horizon_head_output_shape(self):
        """HorizonHead should return (mu_seq, sigma_seq) both shaped (batch, horizon)."""
        head = HorizonHead(latent_size=128, horizon_max=100, d_model=128)
        h_seq = torch.randn(4, 50, 128)  # (batch, seq, d_model)
        horizon = 60
        mu_seq, sigma_seq = head(h_seq, horizon)
        assert mu_seq.shape == (4, 60), f"Expected (4, 60), got {mu_seq.shape}"
        assert sigma_seq.shape == (4, 60), f"Expected (4, 60), got {sigma_seq.shape}"
        assert (sigma_seq > 0).all(), "Sigma should be positive"

    def test_simple_horizon_head_output_shape(self):
        """SimpleHorizonHead should return (mu_seq, sigma_seq) both shaped (batch, horizon)."""
        head = SimpleHorizonHead(latent_size=128, horizon_max=100)
        h_seq = torch.randn(4, 50, 128)  # (batch, seq, d_model)
        horizon = 60
        mu_seq, sigma_seq = head(h_seq, horizon)
        assert mu_seq.shape == (4, 60), f"Expected (4, 60), got {mu_seq.shape}"
        assert sigma_seq.shape == (4, 60), f"Expected (4, 60), got {sigma_seq.shape}"
        assert (sigma_seq > 0).all(), "Sigma should be positive"

    def test_simple_horizon_head_pool_types(self):
        """SimpleHorizonHead should work with all pool types."""
        h_seq = torch.randn(4, 50, 128)
        horizon = 12

        for pool_type in ["mean", "max", "mean+max"]:
            head = SimpleHorizonHead(latent_size=128, horizon_max=48, pool_type=pool_type)
            mu_seq, sigma_seq = head(h_seq, horizon)
            assert mu_seq.shape == (4, 12), f"Pool type {pool_type}: Expected (4, 12), got {mu_seq.shape}"
            assert sigma_seq.shape == (4, 12), f"Pool type {pool_type}: Expected (4, 12), got {sigma_seq.shape}"
            assert (sigma_seq > 0).all(), f"Pool type {pool_type}: Sigma should be positive"

    def test_clt_horizon_head_output_shape(self):
        """CLTHorizonHead should return (mu_seq, sigma_seq) both shaped (batch, horizon)."""
        head = CLTHorizonHead(latent_size=128, hidden=64)
        h_t = torch.randn(4, 128)
        horizon = 60
        mu_seq, sigma_seq = head(h_t, horizon)
        assert mu_seq.shape == (4, 60), f"Expected (4, 60), got {mu_seq.shape}"
        assert sigma_seq.shape == (4, 60), f"Expected (4, 60), got {sigma_seq.shape}"
        assert (sigma_seq > 0).all(), "Sigma should be positive"

    def test_clt_horizon_head_different_horizons(self):
        """CLTHorizonHead should support arbitrary horizon lengths."""
        head = CLTHorizonHead(latent_size=64)
        h_t = torch.randn(2, 64)
        for horizon in [1, 12, 48, 100]:
            mu_seq, sigma_seq = head(h_t, horizon)
            assert mu_seq.shape == (2, horizon)
            assert sigma_seq.shape == (2, horizon)
            assert (sigma_seq > 0).all()

    def test_clt_horizon_head_deterministic(self):
        """CLTHorizonHead should be deterministic (same input → same output)."""
        head = CLTHorizonHead(latent_size=64, hidden=64)
        head.eval()
        h_t = torch.randn(2, 64)
        mu1, sigma1 = head(h_t, 60)
        mu2, sigma2 = head(h_t, 60)
        assert torch.allclose(mu1, mu2), "Deterministic head should reproduce outputs"
        assert torch.allclose(sigma1, sigma2), "Deterministic head should reproduce outputs"

    def test_clt_horizon_head_n_basis(self):
        """CLTHorizonHead should work with different n_basis values."""
        h_t = torch.randn(2, 64)
        for n_basis in [1, 4, 8]:
            head = CLTHorizonHead(latent_size=64, n_basis=n_basis)
            mu_seq, sigma_seq = head(h_t, 12)
            assert mu_seq.shape == (2, 12)
            assert (sigma_seq > 0).all()

    def test_clt_horizon_head_with_simulate_horizon_paths(self):
        """CLTHorizonHead outputs should be compatible with simulate_horizon_paths."""
        head = CLTHorizonHead(latent_size=64)
        h_t = torch.randn(2, 64)
        initial_price = torch.tensor([100.0, 200.0])
        horizon = 12
        n_paths = 50

        mu_seq, sigma_seq = head(h_t, horizon)
        paths = simulate_horizon_paths(initial_price, mu_seq, sigma_seq, n_paths)

        assert paths.shape == (2, 50, 12), f"Expected (2, 50, 12), got {paths.shape}"
        assert (paths > 0).all(), "Prices should be positive"
        assert torch.isfinite(paths).all(), "Paths should not contain NaN or Inf"

    def test_student_t_horizon_head_output_shape(self):
        """StudentTHorizonHead should return (mu_seq, sigma_seq, nu_seq)."""
        head = StudentTHorizonHead(latent_size=128, hidden=64)
        h_t = torch.randn(4, 128)
        horizon = 60
        mu_seq, sigma_seq, nu_seq = head(h_t, horizon)
        assert mu_seq.shape == (4, 60), f"Expected (4, 60), got {mu_seq.shape}"
        assert sigma_seq.shape == (4, 60), f"Expected (4, 60), got {sigma_seq.shape}"
        assert nu_seq.shape == (4, 60), f"Expected (4, 60), got {nu_seq.shape}"
        assert (sigma_seq > 0).all(), "Sigma should be positive"
        assert (nu_seq > 2.0).all(), "Nu should be > 2 for finite variance"

    def test_student_t_horizon_head_different_horizons(self):
        """StudentTHorizonHead should support arbitrary horizon lengths."""
        head = StudentTHorizonHead(latent_size=64)
        h_t = torch.randn(2, 64)
        for horizon in [1, 12, 48, 100]:
            mu_seq, sigma_seq, nu_seq = head(h_t, horizon)
            assert mu_seq.shape == (2, horizon)
            assert sigma_seq.shape == (2, horizon)
            assert nu_seq.shape == (2, horizon)
            assert (sigma_seq > 0).all()
            assert (nu_seq > 2.0).all()

    def test_student_t_horizon_head_with_simulate_t_paths(self):
        """StudentTHorizonHead outputs should be compatible with simulate_t_horizon_paths."""
        head = StudentTHorizonHead(latent_size=64)
        h_t = torch.randn(2, 64)
        initial_price = torch.tensor([100.0, 200.0])
        horizon = 12
        n_paths = 50

        mu_seq, sigma_seq, nu_seq = head(h_t, horizon)
        paths = simulate_t_horizon_paths(initial_price, mu_seq, sigma_seq, nu_seq, n_paths)

        assert paths.shape == (2, 50, 12), f"Expected (2, 50, 12), got {paths.shape}"
        assert (paths > 0).all(), "Prices should be positive"
        assert torch.isfinite(paths).all(), "Paths should not contain NaN or Inf"

    def test_student_t_brownian_walk_smoothness(self):
        """StudentTHorizonHead Brownian walk should produce temporally smooth params."""
        head = StudentTHorizonHead(latent_size=64)
        h_t = torch.randn(4, 64)
        mu_seq, sigma_seq, nu_seq = head(h_t, 60)
        # Step-to-step differences should be small relative to the overall range
        mu_diffs = (mu_seq[:, 1:] - mu_seq[:, :-1]).abs().mean()
        mu_range = mu_seq.max() - mu_seq.min() + 1e-8
        assert mu_diffs / mu_range < 0.5, "Brownian walk should produce smooth trajectories"

    def test_neural_bridge_head_returns_3_values(self):
        """NeuralBridgeHead should return (macro_ret, micro_returns, sigma)."""
        head = NeuralBridgeHead(latent_size=128, micro_steps=12)
        h_t = torch.randn(4, 128)
        result = head(h_t)

        assert len(result) == 3, f"Expected 3 return values, got {len(result)}"
        macro_ret, micro_returns, sigma = result

        assert macro_ret.shape == (4, 1), f"Expected (4, 1), got {macro_ret.shape}"
        assert micro_returns.shape == (4, 12), f"Expected (4, 12), got {micro_returns.shape}"
        assert sigma.shape == (4,), f"Expected (4,), got {sigma.shape}"
        assert (sigma > 0).all(), "Sigma should be positive"

    def test_neural_bridge_head_with_current_price(self):
        """NeuralBridgeHead with current_price should return absolute prices."""
        head = NeuralBridgeHead(latent_size=128, micro_steps=12)
        h_t = torch.randn(4, 128)
        current_price = torch.tensor([100.0, 200.0, 150.0, 180.0])

        macro_ret, micro_path, sigma = head(h_t, current_price=current_price)

        # With current_price, micro_path should be absolute prices
        assert micro_path.shape == (4, 12)
        # Prices should be positive and scaled relative to current_price
        assert (micro_path > 0).all(), "Absolute prices should be positive"


class TestSimulationFunctions:
    """Test simulation functions for correct shapes and numerical stability."""

    def test_simulate_gbm_paths_shape(self):
        """simulate_gbm_paths should return (batch, n_paths, horizon)."""
        initial_price = torch.tensor([100.0, 200.0])
        mu = torch.tensor([0.05, 0.03])
        sigma = torch.tensor([0.1, 0.2])
        horizon = 60
        n_paths = 50

        paths = simulate_gbm_paths(initial_price, mu, sigma, horizon, n_paths)

        assert paths.shape == (2, 50, 60), f"Expected (2, 50, 60), got {paths.shape}"
        assert (paths > 0).all(), "Prices should be positive"

    def test_simulate_horizon_paths_shape(self):
        """simulate_horizon_paths should return (batch, n_paths, horizon)."""
        initial_price = torch.tensor([100.0, 200.0])
        mu_seq = torch.randn(2, 60) * 0.01  # (batch, horizon)
        sigma_seq = torch.abs(torch.randn(2, 60)) * 0.1 + 0.05  # (batch, horizon)
        n_paths = 50

        paths = simulate_horizon_paths(initial_price, mu_seq, sigma_seq, n_paths)

        assert paths.shape == (2, 50, 60), f"Expected (2, 50, 60), got {paths.shape}"
        assert (paths > 0).all(), "Prices should be positive"

    def test_simulate_bridge_paths_shape(self):
        """simulate_bridge_paths should return (batch, n_paths, micro_steps)."""
        initial_price = torch.tensor([100.0, 200.0])
        micro_returns = torch.randn(2, 60) * 0.01  # (batch, micro_steps)
        sigma = torch.tensor([0.1, 0.2])
        n_paths = 50

        paths = simulate_bridge_paths(initial_price, micro_returns, sigma, n_paths)

        assert paths.shape == (2, 50, 60), f"Expected (2, 50, 60), got {paths.shape}"
        assert (paths > 0).all(), "Prices should be positive"

    def test_extreme_log_returns_are_clamped(self):
        """Extreme log returns should be clamped to prevent NaN from exp()."""
        initial_price = torch.tensor([100.0])
        # Set extremely high mu to trigger clamping
        mu = torch.tensor([100.0])  # Would cause exp() overflow without clamping
        sigma = torch.tensor([10.0])
        horizon = 10
        n_paths = 5

        paths = simulate_gbm_paths(initial_price, mu, sigma, horizon, n_paths)

        # Should not contain NaN or Inf
        assert torch.isfinite(paths).all(), "Paths should not contain NaN or Inf"
        # Max possible log return is MAX_LOG_RETURN_CLAMP
        # So max price factor per step is exp(MAX_LOG_RETURN_CLAMP)
        max_expected_step = torch.exp(torch.tensor(MAX_LOG_RETURN_CLAMP))
        # After horizon steps with cumprod, max should be bounded
        assert (paths <= initial_price * (max_expected_step ** horizon)).all()

    def test_negative_log_returns_are_clamped(self):
        """Extremely negative log returns should be clamped."""
        initial_price = torch.tensor([100.0])
        mu = torch.tensor([-100.0])  # Would cause near-zero prices without clamping
        sigma = torch.tensor([0.1])
        horizon = 10
        n_paths = 5

        paths = simulate_gbm_paths(initial_price, mu, sigma, horizon, n_paths)

        # Should not contain NaN or Inf
        assert torch.isfinite(paths).all(), "Paths should not contain NaN or Inf"
        # Prices should remain positive even with extreme negative drift
        assert (paths > 0).all(), "Prices should remain positive"


class TestNumericalStability:
    """Test numerical stability safeguards."""

    def test_zero_sigma_does_not_cause_nan(self):
        """Zero volatility should be handled gracefully (heads add eps)."""
        # Heads add 1e-6 to sigma, so this tests that safeguard
        head = GBMHead(latent_size=4)
        # Force sigma to be very small by manipulating weights
        with torch.no_grad():
            head.sigma_proj.weight.fill_(0.0)
            head.sigma_proj.bias.fill_(-10.0)  # softplus(-10) + 1e-6 ≈ 1e-6

        h_t = torch.randn(2, 4)
        mu, sigma = head(h_t)

        assert (sigma > 0).all(), "Sigma should have minimum threshold"

        # Should be able to simulate without NaN
        initial_price = torch.tensor([100.0, 200.0])
        paths = simulate_gbm_paths(initial_price, mu, sigma, horizon=10, n_paths=5)
        assert torch.isfinite(paths).all()

    def test_large_batch_size(self):
        """Test with large batch size to ensure no memory issues."""
        initial_price = torch.ones(128)  # Large batch
        mu = torch.randn(128) * 0.01
        sigma = torch.abs(torch.randn(128)) * 0.1 + 0.05
        horizon = 60
        n_paths = 100

        paths = simulate_gbm_paths(initial_price, mu, sigma, horizon, n_paths)

        assert paths.shape == (128, 100, 60)
        assert torch.isfinite(paths).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
