import torch

from src.models.registry import registry
from src.models.components.advanced_blocks import LayerNormBlock


def test_layernorm_block_registered():
    """LayerNormBlock should be discoverable via the block registry."""
    cls = registry.get_block("layernormblock")
    assert cls is LayerNormBlock


def test_layernorm_block_preserves_shape():
    """Output shape must match input: (batch, seq, d_model)."""
    block = LayerNormBlock(d_model=32)
    x = torch.randn(4, 10, 32)
    out = block(x)
    assert out.shape == x.shape


def test_layernorm_block_normalizes():
    """Output should be approximately zero-mean, unit-variance along d_model."""
    block = LayerNormBlock(d_model=64)
    x = torch.randn(8, 20, 64) * 5 + 3  # shifted & scaled input
    out = block(x)
    # After LayerNorm the last dim should be ~N(0,1)
    assert out.mean(dim=-1).abs().max() < 0.1
    assert (out.std(dim=-1) - 1.0).abs().max() < 0.1


def test_layernorm_block_between_dlinear():
    """LayerNormBlock should compose with DLinearBlock in a sequential stack."""
    DLinearBlock = registry.get_block("dlinearblock")
    blocks = torch.nn.Sequential(
        DLinearBlock(d_model=32, kernel_size=5),
        LayerNormBlock(d_model=32),
        DLinearBlock(d_model=32, kernel_size=5),
    )
    x = torch.randn(2, 16, 32)
    out = blocks(x)
    assert out.shape == x.shape
