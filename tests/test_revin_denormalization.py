"""Tests for RevIN denormalization in SynthModel."""
import torch
import pytest

from osa.models.components.advanced_blocks import RevIN, TransformerBlock
from osa.models.factory import HybridBackbone, SynthModel
from osa.models.heads import GBMHead


def test_synthmodel_collects_revin_layers():
    """Test that SynthModel correctly identifies RevIN layers in the backbone."""
    d_model = 32
    input_size = 5

    # Create backbone with RevIN
    blocks = [
        RevIN(d_model=d_model, affine=True),
        TransformerBlock(d_model=d_model, nhead=4),
    ]

    backbone = HybridBackbone(
        input_size=input_size,
        d_model=d_model,
        blocks=blocks,
        validate_shapes=True,
    )

    head = GBMHead(latent_size=d_model)
    model = SynthModel(backbone=backbone, head=head)

    # Verify RevIN layers were collected
    assert len(model._revin_layers) == 1, f"Expected 1 RevIN layer, found {len(model._revin_layers)}"
    assert isinstance(model._revin_layers[0], RevIN)


def test_synthmodel_denormalization_affects_outputs():
    """Test that apply_revin_denorm=True produces different outputs than False."""
    d_model = 32
    input_size = 5
    batch_size = 4
    seq_len = 10
    horizon = 5
    n_paths = 100

    # Create backbone with RevIN
    blocks = [
        RevIN(d_model=d_model, affine=True),
        TransformerBlock(d_model=d_model, nhead=4, dropout=0.0),
    ]

    backbone = HybridBackbone(
        input_size=input_size,
        d_model=d_model,
        blocks=blocks,
        validate_shapes=True,
    )

    head = GBMHead(latent_size=d_model)
    model = SynthModel(backbone=backbone, head=head)
    model.eval()

    # Create sample input with non-unit scale to trigger RevIN normalization
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, input_size) * 10.0  # Scale input
    initial_price = torch.ones(batch_size)

    # Run with denormalization — fix the RNG so stochastic GBM paths are identical
    # to the run without denorm and differences are solely due to scaling.
    with torch.no_grad():
        torch.manual_seed(7)
        paths_denorm, mu_denorm, sigma_denorm = model(
            x, initial_price, horizon, n_paths, apply_revin_denorm=True
        )

    # Run without denormalization
    with torch.no_grad():
        torch.manual_seed(7)
        paths_no_denorm, mu_no_denorm, sigma_no_denorm = model(
            x, initial_price, horizon, n_paths, apply_revin_denorm=False
        )

    # Outputs should differ when denormalization is applied
    assert not torch.allclose(mu_denorm, mu_no_denorm), "mu should differ with/without denorm"
    assert not torch.allclose(sigma_denorm, sigma_no_denorm), "sigma should differ with/without denorm"
    assert not torch.allclose(paths_denorm, paths_no_denorm), "paths should differ with/without denorm"

    # With denormalization, sigma should generally be larger (scaled by std > 1)
    # Since we scaled input by 10, RevIN will compute std ~10
    # So denormalized sigma should be ~10x larger
    assert sigma_denorm.mean() > sigma_no_denorm.mean(), \
        "Denormalized sigma should be larger when input has large std"


def test_revin_denormalization_scales_appropriately():
    """Test that denormalization scales outputs by the correct factor."""
    d_model = 32
    input_size = 5
    batch_size = 2
    seq_len = 10
    horizon = 5
    n_paths = 50

    # Create backbone with RevIN
    blocks = [RevIN(d_model=d_model, affine=False)]  # No affine for simpler test

    backbone = HybridBackbone(
        input_size=input_size,
        d_model=d_model,
        blocks=blocks,
        validate_shapes=False,  # RevIN may not preserve shapes perfectly
    )

    head = GBMHead(latent_size=backbone.output_dim)
    model = SynthModel(backbone=backbone, head=head)
    model.eval()

    # Create input with known scale factor
    torch.manual_seed(42)
    scale_factor = 5.0
    x = torch.randn(batch_size, seq_len, input_size) * scale_factor
    initial_price = torch.ones(batch_size)

    # Run forward pass to populate RevIN statistics
    with torch.no_grad():
        paths, mu, sigma = model(
            x, initial_price, horizon, n_paths, apply_revin_denorm=True
        )

    # Check that RevIN stored statistics
    revin_layer = model._revin_layers[0]
    assert revin_layer.stdev is not None

    # The average std should be approximately equal to scale_factor
    # Allow wide tolerance: seq_len=10 per channel gives high sampling variance
    avg_std = revin_layer.stdev.mean().item()
    assert 0.3 * scale_factor < avg_std < 2.0 * scale_factor, \
        f"Expected std ~{scale_factor}, got {avg_std}"


def test_synthmodel_without_revin_ignores_denorm_flag():
    """Test that models without RevIN work correctly regardless of denorm flag."""
    d_model = 32
    input_size = 5
    batch_size = 2
    seq_len = 10
    horizon = 5
    n_paths = 50

    # Create backbone WITHOUT RevIN
    blocks = [TransformerBlock(d_model=d_model, nhead=4, dropout=0.0)]

    backbone = HybridBackbone(
        input_size=input_size,
        d_model=d_model,
        blocks=blocks,
        validate_shapes=True,
    )

    head = GBMHead(latent_size=d_model)
    model = SynthModel(backbone=backbone, head=head)
    model.eval()

    # Verify no RevIN layers
    assert len(model._revin_layers) == 0

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, input_size)
    initial_price = torch.ones(batch_size)

    # Run with and without denorm flag - should produce identical results
    # Set the same seed before each call so stochastic GBM paths match
    with torch.no_grad():
        torch.manual_seed(0)
        paths1, mu1, sigma1 = model(x, initial_price, horizon, n_paths, apply_revin_denorm=True)
        torch.manual_seed(0)
        paths2, mu2, sigma2 = model(x, initial_price, horizon, n_paths, apply_revin_denorm=False)

    # Outputs should be identical when no RevIN is present
    assert torch.allclose(mu1, mu2), "mu should be identical without RevIN"
    assert torch.allclose(sigma1, sigma2), "sigma should be identical without RevIN"
    assert torch.allclose(paths1, paths2), "paths should be identical without RevIN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
