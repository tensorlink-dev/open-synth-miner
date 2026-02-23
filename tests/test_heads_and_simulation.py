"""Tests for head modules and simulation functions."""
from __future__ import annotations

import pytest
import torch

from osa.models.factory import (
    MAX_LOG_RETURN_CLAMP,
    simulate_bridge_paths,
    simulate_gbm_paths,
    simulate_horizon_paths,
    simulate_mixture_paths,
)
from osa.models.heads import (
    GBMHead, HorizonHead, SimpleHorizonHead,
    MixtureDensityHead, VolTermStructureHead,
    NeuralBridgeHead, SDEHead,
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


class TestMixtureDensityHead:
    """Tests for the MixtureDensityHead (K Gaussian mixture components)."""

    def test_output_shape(self):
        """MixtureDensityHead should return (mus, sigmas, weights) shaped (batch, K)."""
        head = MixtureDensityHead(latent_size=64, n_components=3)
        h_t = torch.randn(4, 64)
        mus, sigmas, weights = head(h_t)

        assert mus.shape == (4, 3), f"Expected (4, 3), got {mus.shape}"
        assert sigmas.shape == (4, 3), f"Expected (4, 3), got {sigmas.shape}"
        assert weights.shape == (4, 3), f"Expected (4, 3), got {weights.shape}"
        assert (sigmas > 0).all(), "Sigmas should be positive"
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4)), "Weights should sum to 1"

    @pytest.mark.parametrize("n_components", [2, 3, 5])
    def test_different_n_components(self, n_components):
        """MixtureDensityHead should work with different component counts."""
        head = MixtureDensityHead(latent_size=32, n_components=n_components)
        h_t = torch.randn(2, 32)
        mus, sigmas, weights = head(h_t)
        assert mus.shape == (2, n_components)
        assert sigmas.shape == (2, n_components)
        assert weights.shape == (2, n_components)
        assert (sigmas > 0).all()

    def test_deterministic(self):
        """MixtureDensityHead should be deterministic (same input -> same output)."""
        head = MixtureDensityHead(latent_size=32)
        head.eval()
        h_t = torch.randn(2, 32)
        mus1, sig1, w1 = head(h_t)
        mus2, sig2, w2 = head(h_t)
        assert torch.allclose(mus1, mus2), "Head should be deterministic"
        assert torch.allclose(sig1, sig2), "Head should be deterministic"
        assert torch.allclose(w1, w2), "Head should be deterministic"

    def test_with_simulate_mixture_paths(self):
        """Outputs should be compatible with simulate_mixture_paths."""
        head = MixtureDensityHead(latent_size=64, n_components=3)
        h_t = torch.randn(2, 64)
        initial_price = torch.tensor([100.0, 200.0])
        horizon = 12
        n_paths = 100

        mus, sigmas, weights = head(h_t)
        paths = simulate_mixture_paths(initial_price, mus, sigmas, weights, horizon, n_paths)

        assert paths.shape == (2, 100, 12), f"Expected (2, 100, 12), got {paths.shape}"
        assert (paths > 0).all(), "Prices should be positive"
        assert torch.isfinite(paths).all(), "Paths should not contain NaN or Inf"

    def test_mixture_produces_fat_tails(self):
        """A mixture with different sigmas should produce heavier tails than a single Gaussian."""
        torch.manual_seed(42)
        initial_price = torch.tensor([100.0])
        n_paths = 5000
        horizon = 12

        # Single Gaussian baseline
        mu_single = torch.tensor([0.0])
        sigma_single = torch.tensor([0.15])
        single_paths = simulate_gbm_paths(initial_price, mu_single, sigma_single, horizon, n_paths)

        # Mixture: one low-vol + one high-vol component
        mus = torch.tensor([[0.0, 0.0]])
        sigmas = torch.tensor([[0.05, 0.30]])
        weights = torch.tensor([[0.5, 0.5]])
        mix_paths = simulate_mixture_paths(initial_price, mus, sigmas, weights, horizon, n_paths)

        # Mixture should have higher kurtosis (fatter tails)
        single_returns = (single_paths[0, :, -1] / initial_price - 1)
        mix_returns = (mix_paths[0, :, -1] / initial_price - 1)

        single_kurt = ((single_returns - single_returns.mean()) ** 4).mean() / (single_returns.var() ** 2 + 1e-8)
        mix_kurt = ((mix_returns - mix_returns.mean()) ** 4).mean() / (mix_returns.var() ** 2 + 1e-8)

        assert mix_kurt > single_kurt, (
            f"Mixture kurtosis ({mix_kurt:.2f}) should exceed single Gaussian ({single_kurt:.2f})"
        )

    def test_gradient_flows(self):
        """Gradients should flow through all parameters."""
        head = MixtureDensityHead(latent_size=32, n_components=3)
        h_t = torch.randn(2, 32, requires_grad=True)
        mus, sigmas, weights = head(h_t)
        loss = mus.sum() + sigmas.sum() + weights.sum()
        loss.backward()
        assert h_t.grad is not None, "Gradient should flow to input"
        for name, param in head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_numerical_stability(self):
        """Outputs should be finite with extreme inputs."""
        head = MixtureDensityHead(latent_size=64, n_components=3)
        h_t = torch.randn(4, 64) * 10.0
        mus, sigmas, weights = head(h_t)
        assert torch.isfinite(mus).all(), "mus should be finite"
        assert torch.isfinite(sigmas).all(), "sigmas should be finite"
        assert torch.isfinite(weights).all(), "weights should be finite"


class TestVolTermStructureHead:
    """Tests for the VolTermStructureHead (parametric vol curve)."""

    def test_output_shape(self):
        """VolTermStructureHead should return (mu_seq, sigma_seq) shaped (batch, horizon)."""
        head = VolTermStructureHead(latent_size=64)
        h_t = torch.randn(4, 64)
        mu_seq, sigma_seq = head(h_t, 60)

        assert mu_seq.shape == (4, 60), f"Expected (4, 60), got {mu_seq.shape}"
        assert sigma_seq.shape == (4, 60), f"Expected (4, 60), got {sigma_seq.shape}"
        assert (sigma_seq > 0).all(), "Sigma should be positive"

    def test_different_horizons(self):
        """VolTermStructureHead should support arbitrary horizon lengths."""
        head = VolTermStructureHead(latent_size=32)
        h_t = torch.randn(2, 32)
        for horizon in [1, 12, 48, 100, 288]:
            mu_seq, sigma_seq = head(h_t, horizon)
            assert mu_seq.shape == (2, horizon)
            assert sigma_seq.shape == (2, horizon)
            assert (sigma_seq > 0).all()

    def test_deterministic(self):
        """VolTermStructureHead should be deterministic."""
        head = VolTermStructureHead(latent_size=32)
        head.eval()
        h_t = torch.randn(2, 32)
        mu1, sig1 = head(h_t, 24)
        mu2, sig2 = head(h_t, 24)
        assert torch.allclose(mu1, mu2), "Head should be deterministic"
        assert torch.allclose(sig1, sig2), "Head should be deterministic"

    def test_sigma_monotonic(self):
        """With positive vol_slope, sigma should increase over the horizon."""
        head = VolTermStructureHead(latent_size=32)
        # Force positive vol_slope by manipulating param_proj bias
        with torch.no_grad():
            head.param_proj.weight.fill_(0.0)
            # [mu_0, sigma_0_pre, vol_slope_pre, drift_slope_pre]
            head.param_proj.bias.copy_(torch.tensor([0.0, 0.0, 2.0, 0.0]))
        h_t = torch.zeros(1, 32)
        _, sigma_seq = head(h_t, 24)
        # sigma should be monotonically increasing
        diffs = sigma_seq[0, 1:] - sigma_seq[0, :-1]
        assert (diffs >= 0).all(), "Positive vol_slope should give increasing sigma"

    def test_vol_slope_bounded(self):
        """Vol slope should be bounded by max_vol_slope."""
        head = VolTermStructureHead(latent_size=32, max_vol_slope=2.0)
        h_t = torch.randn(4, 32) * 100.0  # extreme inputs
        _, sigma_seq = head(h_t, 60)
        # sigma ratio between last and first step should be bounded by exp(max_vol_slope)
        import math
        ratio = sigma_seq[:, -1] / sigma_seq[:, 0]
        max_ratio = math.exp(2.0)
        assert (ratio <= max_ratio + 0.01).all(), f"Vol ratio should be <= exp(2.0), got {ratio.max():.2f}"

    def test_with_simulate_horizon_paths(self):
        """Outputs should be compatible with simulate_horizon_paths."""
        head = VolTermStructureHead(latent_size=64)
        h_t = torch.randn(2, 64)
        initial_price = torch.tensor([100.0, 200.0])
        horizon = 12
        n_paths = 50

        mu_seq, sigma_seq = head(h_t, horizon)
        paths = simulate_horizon_paths(initial_price, mu_seq, sigma_seq, n_paths)

        assert paths.shape == (2, 50, 12), f"Expected (2, 50, 12), got {paths.shape}"
        assert (paths > 0).all(), "Prices should be positive"
        assert torch.isfinite(paths).all(), "Paths should not contain NaN or Inf"

    def test_gradient_flows(self):
        """Gradients should flow through all parameters."""
        head = VolTermStructureHead(latent_size=32)
        h_t = torch.randn(2, 32, requires_grad=True)
        mu_seq, sigma_seq = head(h_t, 12)
        loss = mu_seq.sum() + sigma_seq.sum()
        loss.backward()
        assert h_t.grad is not None, "Gradient should flow to input"
        for name, param in head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_numerical_stability(self):
        """Outputs should be finite with extreme inputs."""
        head = VolTermStructureHead(latent_size=64)
        h_t = torch.randn(4, 64) * 10.0
        mu_seq, sigma_seq = head(h_t, 48)
        assert torch.isfinite(mu_seq).all(), "mu should be finite"
        assert torch.isfinite(sigma_seq).all(), "sigma should be finite"

    def test_long_horizon_288_stability(self):
        """Sigma should remain bounded at H=288 (24h @ 5min)."""
        head = VolTermStructureHead(latent_size=64)
        h_t = torch.randn(4, 64)
        mu_seq, sigma_seq = head(h_t, 288)
        assert sigma_seq.shape == (4, 288)
        assert torch.isfinite(sigma_seq).all(), "sigma not finite at H=288"
        assert (sigma_seq > 0).all(), "sigma not positive at H=288"
        # With max_vol_slope=2.0, sigma can grow at most exp(2)≈7.4x
        assert sigma_seq.max() < 100.0, f"sigma too large at H=288: {sigma_seq.max():.2f}"

    def test_degenerates_to_constant_with_zero_slopes(self):
        """With zero slopes, should produce constant mu and sigma across horizon."""
        head = VolTermStructureHead(latent_size=32)
        with torch.no_grad():
            head.param_proj.weight.fill_(0.0)
            head.param_proj.bias.copy_(torch.tensor([0.05, 0.5, 0.0, 0.0]))
        h_t = torch.zeros(1, 32)
        mu_seq, sigma_seq = head(h_t, 24)
        # mu should be constant
        assert torch.allclose(mu_seq[0, 0:1].expand(24), mu_seq[0], atol=1e-5), \
            "Zero drift_slope should give constant mu"
        # sigma should be constant
        assert torch.allclose(sigma_seq[0, 0:1].expand(24), sigma_seq[0], atol=1e-5), \
            "Zero vol_slope should give constant sigma"


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

    def test_simulate_mixture_paths_shape(self):
        """simulate_mixture_paths should return (batch, n_paths, horizon)."""
        initial_price = torch.tensor([100.0, 200.0])
        mus = torch.tensor([[0.05, -0.02, 0.1], [0.03, 0.0, 0.08]])
        sigmas = torch.tensor([[0.1, 0.05, 0.3], [0.2, 0.1, 0.4]])
        weights = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        horizon = 60
        n_paths = 50

        paths = simulate_mixture_paths(initial_price, mus, sigmas, weights, horizon, n_paths)

        assert paths.shape == (2, 50, 60), f"Expected (2, 50, 60), got {paths.shape}"
        assert (paths > 0).all(), "Prices should be positive"
        assert torch.isfinite(paths).all(), "Paths should not contain NaN or Inf"

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
