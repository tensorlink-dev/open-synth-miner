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
    StudentTHorizonHead, ProbabilisticHorizonHead, HorizonHeadUnification,
    GaussianSpectralHead,
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


class TestProbabilisticHorizonHead:
    """Tests for the unified ProbabilisticHorizonHead (spectral/brownian/hybrid/hybrid_ou)."""

    ALL_MODES = ["spectral", "brownian", "hybrid", "hybrid_ou"]

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_output_shape(self, mode):
        """All modes should return (mu_seq, sigma_seq, nu_seq) shaped (batch, horizon)."""
        head = ProbabilisticHorizonHead(latent_size=64, hidden_dim=32, mode=mode)
        h_t = torch.randn(4, 64)
        mu_seq, sigma_seq, nu_seq = head(h_t, 60)

        assert mu_seq.shape == (4, 60), f"mu shape mismatch for mode={mode}"
        assert sigma_seq.shape == (4, 60), f"sigma shape mismatch for mode={mode}"
        assert nu_seq.shape == (4, 60), f"nu shape mismatch for mode={mode}"
        assert (sigma_seq > 0).all(), f"Sigma should be positive for mode={mode}"
        assert (nu_seq >= 2.1).all(), f"Nu should be >= 2.1 for mode={mode}"
        assert (nu_seq <= 30.1).all(), f"Nu should be <= 30.1 for mode={mode}"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_different_horizons(self, mode):
        """All modes should support arbitrary horizon lengths."""
        head = ProbabilisticHorizonHead(latent_size=32, mode=mode)
        h_t = torch.randn(2, 32)
        for horizon in [1, 12, 48, 100]:
            mu_seq, sigma_seq, nu_seq = head(h_t, horizon)
            assert mu_seq.shape == (2, horizon)
            assert sigma_seq.shape == (2, horizon)
            assert nu_seq.shape == (2, horizon)
            assert (sigma_seq > 0).all()

    def test_spectral_is_deterministic(self):
        """Spectral mode should be deterministic (same input -> same output)."""
        head = ProbabilisticHorizonHead(latent_size=32, mode="spectral")
        head.eval()
        h_t = torch.randn(2, 32)
        mu1, sig1, nu1 = head(h_t, 24)
        mu2, sig2, nu2 = head(h_t, 24)
        assert torch.allclose(mu1, mu2), "Spectral mode should be deterministic"
        assert torch.allclose(sig1, sig2), "Spectral mode should be deterministic"
        assert torch.allclose(nu1, nu2), "Spectral mode should be deterministic"

    @pytest.mark.parametrize("mode", ["brownian", "hybrid", "hybrid_ou"])
    def test_stochastic_modes_are_stochastic(self, mode):
        """Stochastic modes should produce different outputs across calls."""
        head = ProbabilisticHorizonHead(latent_size=32, mode=mode)
        head.eval()
        h_t = torch.randn(2, 32)
        mu1, _, _ = head(h_t, 24)
        mu2, _, _ = head(h_t, 24)
        assert not torch.allclose(mu1, mu2, atol=1e-6), f"{mode} mode should be stochastic"

    def test_brownian_walk_smoothness(self):
        """Brownian walk should produce temporally smooth parameter trajectories."""
        head = ProbabilisticHorizonHead(latent_size=64, mode="brownian")
        h_t = torch.randn(4, 64)
        mu_seq, _, _ = head(h_t, 60)
        mu_diffs = (mu_seq[:, 1:] - mu_seq[:, :-1]).abs().mean()
        mu_range = mu_seq.max() - mu_seq.min() + 1e-8
        assert mu_diffs / mu_range < 0.5, "Brownian walk should produce smooth trajectories"

    @pytest.mark.parametrize("n_basis", [1, 4, 8, 16])
    def test_different_n_basis(self, n_basis):
        """Spectral/hybrid modes should work with different n_basis values."""
        for mode in ["spectral", "hybrid", "hybrid_ou"]:
            head = ProbabilisticHorizonHead(latent_size=32, mode=mode, n_basis=n_basis)
            h_t = torch.randn(2, 32)
            mu_seq, sigma_seq, nu_seq = head(h_t, 12)
            assert mu_seq.shape == (2, 12)
            assert (sigma_seq > 0).all()

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="mode must be one of"):
            ProbabilisticHorizonHead(latent_size=32, mode="invalid")

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_with_simulate_t_horizon_paths(self, mode):
        """Outputs should be compatible with simulate_t_horizon_paths."""
        head = ProbabilisticHorizonHead(latent_size=64, mode=mode)
        h_t = torch.randn(2, 64)
        initial_price = torch.tensor([100.0, 200.0])
        horizon = 12
        n_paths = 50

        mu_seq, sigma_seq, nu_seq = head(h_t, horizon)
        paths = simulate_t_horizon_paths(initial_price, mu_seq, sigma_seq, nu_seq, n_paths)

        assert paths.shape == (2, 50, 12), f"Expected (2, 50, 12), got {paths.shape}"
        assert (paths > 0).all(), "Prices should be positive"
        assert torch.isfinite(paths).all(), "Paths should not contain NaN or Inf"

    def test_spectral_only_has_basis_weights(self):
        """Spectral mode should have basis_weights but not param_proj."""
        head = ProbabilisticHorizonHead(latent_size=32, mode="spectral")
        assert hasattr(head, "basis_weights")
        assert not hasattr(head, "param_proj")

    def test_brownian_only_has_param_proj(self):
        """Brownian mode should have param_proj but not basis_weights."""
        head = ProbabilisticHorizonHead(latent_size=32, mode="brownian")
        assert hasattr(head, "param_proj")
        assert not hasattr(head, "basis_weights")

    def test_hybrid_has_both(self):
        """Hybrid mode should have both basis_weights and param_proj."""
        head = ProbabilisticHorizonHead(latent_size=32, mode="hybrid")
        assert hasattr(head, "basis_weights")
        assert hasattr(head, "param_proj")
        assert hasattr(head, "mix_logit")

    def test_hybrid_ou_has_reversion(self):
        """hybrid_ou mode should have reversion_logit and mix_logit."""
        head = ProbabilisticHorizonHead(latent_size=32, mode="hybrid_ou")
        assert hasattr(head, "basis_weights")
        assert hasattr(head, "param_proj")
        assert hasattr(head, "mix_logit")
        assert hasattr(head, "reversion_logit")

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_numerical_stability(self, mode):
        """Outputs should be finite for all modes."""
        head = ProbabilisticHorizonHead(latent_size=64, mode=mode)
        # Test with extreme inputs
        h_t = torch.randn(4, 64) * 10.0
        mu_seq, sigma_seq, nu_seq = head(h_t, 48)
        assert torch.isfinite(mu_seq).all(), f"mu has non-finite values for mode={mode}"
        assert torch.isfinite(sigma_seq).all(), f"sigma has non-finite values for mode={mode}"
        assert torch.isfinite(nu_seq).all(), f"nu has non-finite values for mode={mode}"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_long_horizon_288_stability(self, mode):
        """All modes should produce bounded sigma at H=288 (24h @ 5min)."""
        head = ProbabilisticHorizonHead(latent_size=64, mode=mode)
        h_t = torch.randn(4, 64)
        mu_seq, sigma_seq, nu_seq = head(h_t, 288)

        assert sigma_seq.shape == (4, 288)
        assert torch.isfinite(sigma_seq).all(), f"sigma not finite for mode={mode} at H=288"
        assert (sigma_seq > 0).all(), f"sigma not positive for mode={mode} at H=288"
        # With clamping, sigma should stay bounded — the -0.5*sigma^2 drift
        # must not collapse paths to zero over 288 steps
        assert sigma_seq.max() < 10.0, (
            f"sigma too large for mode={mode} at H=288: max={sigma_seq.max():.2f}"
        )

    def test_hybrid_mix_gate_initialises_small(self):
        """Hybrid mix gate should start near zero so training begins ~spectral."""
        head = ProbabilisticHorizonHead(latent_size=32, mode="hybrid")
        mix = torch.sigmoid(head.mix_logit).item()
        assert mix < 0.2, f"mix gate should init small, got {mix:.3f}"

    def test_ou_reversion_bounds_variance(self):
        """OU mode should have lower endpoint variance than pure Brownian."""
        torch.manual_seed(42)
        h_t = torch.randn(8, 64)
        horizon = 288

        brownian_head = ProbabilisticHorizonHead(latent_size=64, mode="brownian")
        ou_head = ProbabilisticHorizonHead(latent_size=64, mode="hybrid_ou")
        # Copy shared weights so comparison is fair
        ou_head.norm.load_state_dict(brownian_head.norm.state_dict())
        ou_head.net.load_state_dict(brownian_head.net.state_dict())
        ou_head.param_proj.load_state_dict(brownian_head.param_proj.state_dict())

        # Run many times and measure variance of the endpoint sigma
        brownian_sigmas = []
        ou_sigmas = []
        for _ in range(20):
            _, sig_b, _ = brownian_head(h_t, horizon)
            _, sig_ou, _ = ou_head(h_t, horizon)
            brownian_sigmas.append(sig_b[:, -1].detach())
            ou_sigmas.append(sig_ou[:, -1].detach())

        var_brownian = torch.stack(brownian_sigmas).var(dim=0).mean()
        var_ou = torch.stack(ou_sigmas).var(dim=0).mean()
        assert var_ou < var_brownian, (
            f"OU endpoint variance ({var_ou:.4f}) should be less than "
            f"Brownian ({var_brownian:.4f})"
        )


class TestHorizonHeadUnification:
    """Tests for the three-way unified HorizonHeadUnification (spectral + fractal + DC)."""

    @pytest.mark.parametrize("mode", ["spectral", "fractal", "hybrid"])
    def test_output_shape(self, mode):
        """All modes should return (mu_seq, sigma_seq, nu_seq) shaped (batch, horizon)."""
        head = HorizonHeadUnification(latent_size=64, hidden_dim=32, n_basis=8)
        h_t = torch.randn(4, 64)
        mu_seq, sigma_seq, nu_seq = head(h_t, 60, mode=mode)

        assert mu_seq.shape == (4, 60), f"mu shape mismatch for mode={mode}"
        assert sigma_seq.shape == (4, 60), f"sigma shape mismatch for mode={mode}"
        assert nu_seq.shape == (4, 60), f"nu shape mismatch for mode={mode}"
        assert (sigma_seq > 0).all(), f"Sigma should be positive for mode={mode}"
        assert (nu_seq >= 2.1).all(), f"Nu should be >= 2.1 for mode={mode}"
        assert (nu_seq <= 30.1).all(), f"Nu should be <= 30.1 for mode={mode}"

    @pytest.mark.parametrize("mode", ["spectral", "fractal", "hybrid"])
    def test_different_horizons(self, mode):
        """All modes should support arbitrary horizon lengths."""
        head = HorizonHeadUnification(latent_size=32, n_basis=6)
        h_t = torch.randn(2, 32)
        for horizon in [1, 12, 48, 100]:
            mu_seq, sigma_seq, nu_seq = head(h_t, horizon, mode=mode)
            assert mu_seq.shape == (2, horizon)
            assert sigma_seq.shape == (2, horizon)
            assert nu_seq.shape == (2, horizon)
            assert (sigma_seq > 0).all()

    def test_spectral_is_deterministic(self):
        """Spectral mode should be deterministic (same input -> same output)."""
        head = HorizonHeadUnification(latent_size=32)
        head.eval()
        h_t = torch.randn(2, 32)
        mu1, sig1, nu1 = head(h_t, 24, mode="spectral")
        mu2, sig2, nu2 = head(h_t, 24, mode="spectral")
        assert torch.allclose(mu1, mu2), "Spectral mode should be deterministic"
        assert torch.allclose(sig1, sig2), "Spectral mode should be deterministic"
        assert torch.allclose(nu1, nu2), "Spectral mode should be deterministic"

    def test_fractal_is_stochastic(self):
        """Fractal mode should produce different outputs across calls."""
        head = HorizonHeadUnification(latent_size=32)
        head.eval()
        h_t = torch.randn(2, 32)
        mu1, _, _ = head(h_t, 24, mode="fractal")
        mu2, _, _ = head(h_t, 24, mode="fractal")
        assert not torch.allclose(mu1, mu2, atol=1e-6), "Fractal mode should be stochastic"

    def test_hybrid_is_stochastic(self):
        """Hybrid mode should be stochastic due to fractal component."""
        head = HorizonHeadUnification(latent_size=32)
        head.eval()
        h_t = torch.randn(2, 32)
        mu1, _, _ = head(h_t, 24, mode="hybrid")
        mu2, _, _ = head(h_t, 24, mode="hybrid")
        assert not torch.allclose(mu1, mu2, atol=1e-6), "Hybrid mode should be stochastic"

    def test_fractal_walk_smoothness(self):
        """Fractal Brownian walk should produce temporally smooth trajectories."""
        head = HorizonHeadUnification(latent_size=64)
        h_t = torch.randn(4, 64)
        mu_seq, _, _ = head(h_t, 60, mode="fractal")
        mu_diffs = (mu_seq[:, 1:] - mu_seq[:, :-1]).abs().mean()
        mu_range = mu_seq.max() - mu_seq.min() + 1e-8
        assert mu_diffs / mu_range < 0.5, "Brownian walk should produce smooth trajectories"

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        head = HorizonHeadUnification(latent_size=32)
        h_t = torch.randn(2, 32)
        with pytest.raises(ValueError, match="mode must be one of"):
            head(h_t, 12, mode="invalid")

    @pytest.mark.parametrize("n_basis", [1, 4, 12, 16])
    def test_different_n_basis(self, n_basis):
        """Spectral/hybrid modes should work with different n_basis values."""
        head = HorizonHeadUnification(latent_size=32, n_basis=n_basis)
        h_t = torch.randn(2, 32)
        for mode in ["spectral", "hybrid"]:
            mu_seq, sigma_seq, nu_seq = head(h_t, 12, mode=mode)
            assert mu_seq.shape == (2, 12)
            assert (sigma_seq > 0).all()

    @pytest.mark.parametrize("mode", ["spectral", "fractal", "hybrid"])
    def test_with_simulate_t_horizon_paths(self, mode):
        """Outputs should be compatible with simulate_t_horizon_paths."""
        head = HorizonHeadUnification(latent_size=64)
        h_t = torch.randn(2, 64)
        initial_price = torch.tensor([100.0, 200.0])
        horizon = 12
        n_paths = 50

        mu_seq, sigma_seq, nu_seq = head(h_t, horizon, mode=mode)
        paths = simulate_t_horizon_paths(initial_price, mu_seq, sigma_seq, nu_seq, n_paths)

        assert paths.shape == (2, 50, 12), f"Expected (2, 50, 12), got {paths.shape}"
        assert (paths > 0).all(), "Prices should be positive"
        assert torch.isfinite(paths).all(), "Paths should not contain NaN or Inf"

    @pytest.mark.parametrize("mode", ["spectral", "fractal", "hybrid"])
    def test_numerical_stability(self, mode):
        """Outputs should be finite for all modes with extreme inputs."""
        head = HorizonHeadUnification(latent_size=64)
        h_t = torch.randn(4, 64) * 10.0
        mu_seq, sigma_seq, nu_seq = head(h_t, 48, mode=mode)
        assert torch.isfinite(mu_seq).all(), f"mu has non-finite values for mode={mode}"
        assert torch.isfinite(sigma_seq).all(), f"sigma has non-finite values for mode={mode}"
        assert torch.isfinite(nu_seq).all(), f"nu has non-finite values for mode={mode}"

    def test_mode_is_per_call(self):
        """Same head instance should support different modes per forward call."""
        head = HorizonHeadUnification(latent_size=32)
        h_t = torch.randn(2, 32)
        # All three modes should work on the same instance
        mu_s, sig_s, nu_s = head(h_t, 12, mode="spectral")
        mu_f, sig_f, nu_f = head(h_t, 12, mode="fractal")
        mu_h, sig_h, nu_h = head(h_t, 12, mode="hybrid")
        # All should have valid shapes
        for mu in [mu_s, mu_f, mu_h]:
            assert mu.shape == (2, 12)

    def test_dc_offset_always_present(self):
        """Static DC offset should contribute in all modes."""
        head = HorizonHeadUnification(latent_size=32)
        assert hasattr(head, "static_proj"), "Should have static_proj for DC offset"
        assert hasattr(head, "spectral_proj"), "Should always have spectral_proj"
        assert hasattr(head, "diffusion_proj"), "Should always have diffusion_proj"


class TestGaussianSpectralHead:
    """Tests for the Gaussian spectral head (spectral basis → Gaussian simulation)."""

    ALL_MODES = ["spectral", "fractal", "hybrid"]

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_output_shape(self, mode):
        """All modes should return (mu_seq, sigma_seq) shaped (batch, horizon)."""
        head = GaussianSpectralHead(latent_size=64, hidden_dim=32, mode=mode)
        h_t = torch.randn(4, 64)
        mu_seq, sigma_seq = head(h_t, 60)

        assert mu_seq.shape == (4, 60), f"mu shape mismatch for mode={mode}"
        assert sigma_seq.shape == (4, 60), f"sigma shape mismatch for mode={mode}"
        assert (sigma_seq > 0).all(), f"Sigma should be positive for mode={mode}"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_returns_two_values_not_three(self, mode):
        """GaussianSpectralHead should return exactly 2 values (no nu)."""
        head = GaussianSpectralHead(latent_size=32, mode=mode)
        h_t = torch.randn(2, 32)
        result = head(h_t, 12)
        assert len(result) == 2, f"Expected 2 return values (mu, sigma), got {len(result)}"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_different_horizons(self, mode):
        """All modes should support arbitrary horizon lengths."""
        head = GaussianSpectralHead(latent_size=32, mode=mode)
        h_t = torch.randn(2, 32)
        for horizon in [1, 12, 48, 100, 288]:
            mu_seq, sigma_seq = head(h_t, horizon)
            assert mu_seq.shape == (2, horizon)
            assert sigma_seq.shape == (2, horizon)
            assert (sigma_seq > 0).all()

    def test_spectral_is_deterministic(self):
        """Spectral mode should be deterministic (same input -> same output)."""
        head = GaussianSpectralHead(latent_size=32, mode="spectral")
        head.eval()
        h_t = torch.randn(2, 32)
        mu1, sig1 = head(h_t, 24)
        mu2, sig2 = head(h_t, 24)
        assert torch.allclose(mu1, mu2), "Spectral mode should be deterministic"
        assert torch.allclose(sig1, sig2), "Spectral mode should be deterministic"

    def test_fractal_is_stochastic(self):
        """Fractal mode should produce different outputs across calls."""
        head = GaussianSpectralHead(latent_size=32, mode="fractal")
        head.eval()
        h_t = torch.randn(2, 32)
        mu1, _ = head(h_t, 24)
        mu2, _ = head(h_t, 24)
        assert not torch.allclose(mu1, mu2, atol=1e-6), "Fractal mode should be stochastic"

    def test_hybrid_is_stochastic(self):
        """Hybrid mode should be stochastic due to fractal component."""
        head = GaussianSpectralHead(latent_size=32, mode="hybrid")
        head.eval()
        h_t = torch.randn(2, 32)
        mu1, _ = head(h_t, 24)
        mu2, _ = head(h_t, 24)
        assert not torch.allclose(mu1, mu2, atol=1e-6), "Hybrid mode should be stochastic"

    def test_fractal_walk_smoothness(self):
        """Fractal Brownian walk should produce temporally smooth trajectories."""
        head = GaussianSpectralHead(latent_size=64, mode="fractal")
        h_t = torch.randn(4, 64)
        mu_seq, _ = head(h_t, 60)
        mu_diffs = (mu_seq[:, 1:] - mu_seq[:, :-1]).abs().mean()
        mu_range = mu_seq.max() - mu_seq.min() + 1e-8
        assert mu_diffs / mu_range < 0.5, "Brownian walk should produce smooth trajectories"

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="mode must be one of"):
            GaussianSpectralHead(latent_size=32, mode="invalid")

    @pytest.mark.parametrize("n_basis", [1, 4, 12, 16])
    def test_different_n_basis(self, n_basis):
        """Spectral/hybrid modes should work with different n_basis values."""
        for mode in ["spectral", "hybrid"]:
            head = GaussianSpectralHead(latent_size=32, n_basis=n_basis, mode=mode)
            h_t = torch.randn(2, 32)
            mu_seq, sigma_seq = head(h_t, 12)
            assert mu_seq.shape == (2, 12)
            assert (sigma_seq > 0).all()

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_with_simulate_horizon_paths(self, mode):
        """Outputs should be compatible with simulate_horizon_paths (Gaussian)."""
        head = GaussianSpectralHead(latent_size=64, mode=mode)
        h_t = torch.randn(2, 64)
        initial_price = torch.tensor([100.0, 200.0])
        horizon = 12
        n_paths = 50

        mu_seq, sigma_seq = head(h_t, horizon)
        paths = simulate_horizon_paths(initial_price, mu_seq, sigma_seq, n_paths)

        assert paths.shape == (2, 50, 12), f"Expected (2, 50, 12), got {paths.shape}"
        assert (paths > 0).all(), "Prices should be positive"
        assert torch.isfinite(paths).all(), "Paths should not contain NaN or Inf"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_numerical_stability(self, mode):
        """Outputs should be finite for all modes with extreme inputs."""
        head = GaussianSpectralHead(latent_size=64, mode=mode)
        h_t = torch.randn(4, 64) * 10.0
        mu_seq, sigma_seq = head(h_t, 48)
        assert torch.isfinite(mu_seq).all(), f"mu has non-finite values for mode={mode}"
        assert torch.isfinite(sigma_seq).all(), f"sigma has non-finite values for mode={mode}"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_long_horizon_288_stability(self, mode):
        """All modes should produce bounded sigma at H=288 (24h @ 5min)."""
        head = GaussianSpectralHead(latent_size=64, mode=mode)
        h_t = torch.randn(4, 64)
        mu_seq, sigma_seq = head(h_t, 288)

        assert sigma_seq.shape == (4, 288)
        assert torch.isfinite(sigma_seq).all(), f"sigma not finite for mode={mode} at H=288"
        assert (sigma_seq > 0).all(), f"sigma not positive for mode={mode} at H=288"
        assert sigma_seq.max() < 10.0, (
            f"sigma too large for mode={mode} at H=288: max={sigma_seq.max():.2f}"
        )

    def test_has_no_nu_related_params(self):
        """GaussianSpectralHead should have no nu-related projections."""
        head = GaussianSpectralHead(latent_size=32)
        assert hasattr(head, "static_proj"), "Should have static_proj for DC offset"
        assert hasattr(head, "spectral_proj"), "Should have spectral_proj"
        assert hasattr(head, "diffusion_proj"), "Should have diffusion_proj"
        # static_proj output should be 2 (mu, sigma), not 3
        assert head.static_proj.out_features == 2
        assert head.diffusion_proj.out_features == 2

    def test_dc_offset_always_present(self):
        """Static DC offset should contribute in all modes."""
        head = GaussianSpectralHead(latent_size=32, mode="spectral")
        h_t = torch.randn(2, 32)
        # Zero out spectral weights to isolate DC offset
        with torch.no_grad():
            head.spectral_proj.weight.fill_(0.0)
            head.spectral_proj.bias.fill_(0.0)
        mu_seq, sigma_seq = head(h_t, 12)
        # With spectral zeroed, each step should be identical (only DC)
        assert torch.allclose(mu_seq[:, 0:1].expand_as(mu_seq), mu_seq, atol=1e-5), \
            "With spectral zeroed, mu should be constant (DC only)"

    def test_gradient_flows(self):
        """Gradients should flow through all parameters."""
        head = GaussianSpectralHead(latent_size=32, mode="hybrid")
        h_t = torch.randn(2, 32, requires_grad=True)
        mu_seq, sigma_seq = head(h_t, 12)
        loss = mu_seq.sum() + sigma_seq.sum()
        loss.backward()
        assert h_t.grad is not None, "Gradient should flow to input"
        for name, param in head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


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
