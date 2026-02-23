"""Tests for HybridBackbone and LayerNorm insertion behaviour."""
import pathlib

import pytest
import torch

# Guard optional-but-expected dependencies so the test module is skipped cleanly
# instead of failing with an ImportError when hydra/omegaconf are absent.
pytest.importorskip("hydra", reason="hydra not installed")
pytest.importorskip("omegaconf", reason="omegaconf not installed")

from hydra.utils import instantiate
from omegaconf import OmegaConf

from osa.models.components.advanced_blocks import (
    ChannelRejoin,
    DLinearBlock,
    FlexiblePatchEmbed,
    LayerNormBlock,
)
from osa.models.factory import HybridBackbone

# Resolve config paths relative to this file so tests pass regardless of
# the working directory from which pytest is invoked.
_CONFIGS = pathlib.Path(__file__).parent.parent / "configs" / "model"


def test_hybrid_backbone_infers_output_dim_with_input_projection():
    cfg = OmegaConf.load(_CONFIGS / "hybrid_v2.yaml")
    backbone = instantiate(cfg.model.backbone)

    sample = torch.randn(2, 3, cfg.model.backbone.input_size)
    output = backbone(sample)

    assert backbone.output_dim == cfg.model.backbone.d_model
    assert output.shape == (2, cfg.model.backbone.d_model)


def test_hybrid_backbone_automatic_layernorm_insertion():
    """insert_layernorm=True should interleave LayerNormBlocks between the original blocks."""
    cfg = OmegaConf.load(_CONFIGS / "hybrid_with_layernorm.yaml")
    backbone = instantiate(cfg.model.backbone)

    # Behavioural assertion: LayerNormBlocks should be present between blocks.
    # For 3 original blocks, 2 LayerNorm layers should be inserted.
    layernorm_indices = [
        i for i, layer in enumerate(backbone.layers) if isinstance(layer, LayerNormBlock)
    ]
    assert len(layernorm_indices) == 2, (
        f"Expected 2 LayerNormBlocks for a 3-block config, "
        f"got {len(layernorm_indices)} at indices {layernorm_indices}"
    )
    # LayerNorms should not be at the very start or end.
    assert layernorm_indices[0] > 0
    assert layernorm_indices[-1] < len(backbone.layers) - 1

    sample = torch.randn(2, 10, cfg.model.backbone.input_size)
    output = backbone(sample)
    assert output.shape == (2, cfg.model.backbone.d_model)


def test_hybrid_backbone_manual_layernorm_insertion():
    """Manually placed LayerNormBlocks should be preserved exactly as configured."""
    cfg = OmegaConf.load(_CONFIGS / "hybrid_manual_layernorm.yaml")
    backbone = instantiate(cfg.model.backbone)

    layernorm_indices = [
        i for i, layer in enumerate(backbone.layers) if isinstance(layer, LayerNormBlock)
    ]
    assert len(layernorm_indices) == 2, (
        f"Expected 2 manual LayerNormBlocks, got {len(layernorm_indices)}"
    )

    sample = torch.randn(2, 10, cfg.model.backbone.input_size)
    output = backbone(sample)
    assert output.shape == (2, cfg.model.backbone.d_model)


def test_hybrid_backbone_layernorm_block_shape_preservation():
    """LayerNormBlocks should not alter the (batch, seq, d_model) shape through the sequence."""
    cfg = OmegaConf.load(_CONFIGS / "hybrid_with_layernorm.yaml")
    backbone = instantiate(cfg.model.backbone)

    batch_size = 4
    seq_len = 15
    sample = torch.randn(batch_size, seq_len, cfg.model.backbone.input_size)

    output_seq = backbone.forward_sequence(sample)
    assert output_seq.shape == (batch_size, seq_len, cfg.model.backbone.d_model)

    output = backbone(sample)
    assert output.shape == (batch_size, cfg.model.backbone.d_model)


def test_skip_input_proj_with_patch_embed_and_channel_rejoin():
    """FlexiblePatchEmbed + ChannelRejoin with skip_input_proj produces correct shapes."""
    d_model = 32
    in_channels = 3
    patch_len = 12
    stride = 12

    blocks = [
        FlexiblePatchEmbed(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            in_channels=in_channels,
            channel_independence=True,
        ),
        DLinearBlock(d_model=d_model, kernel_size=5),
        ChannelRejoin(num_channels=in_channels, mode="mean"),
    ]

    backbone = HybridBackbone(
        input_size=in_channels,
        d_model=d_model,
        blocks=blocks,
        skip_input_proj=True,
        validate_shapes=False,
    )

    batch, seq = 4, 64
    x = torch.randn(batch, seq, in_channels)
    out = backbone(x)
    assert out.shape == (batch, d_model), f"Expected ({batch}, {d_model}), got {out.shape}"

    out_seq = backbone.forward_sequence(x)
    assert out_seq.ndim == 3
    assert out_seq.shape[0] == batch
    assert out_seq.shape[2] == d_model


def test_skip_input_proj_with_flatten_rejoin():
    """ChannelRejoin mode='flatten' produces (batch, seq, channels * d_model)."""
    d_model = 16
    in_channels = 5
    patch_len = 8
    stride = 8

    blocks = [
        FlexiblePatchEmbed(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            in_channels=in_channels,
            channel_independence=True,
        ),
        ChannelRejoin(num_channels=in_channels, mode="flatten"),
    ]

    backbone = HybridBackbone(
        input_size=in_channels,
        d_model=d_model,
        blocks=blocks,
        skip_input_proj=True,
        validate_shapes=False,
    )

    batch, seq = 2, 64
    x = torch.randn(batch, seq, in_channels)
    out = backbone(x)
    assert out.shape == (batch, in_channels * d_model)


def test_skip_input_proj_false_is_backward_compatible():
    """Default skip_input_proj=False preserves existing behavior."""
    d_model = 32
    blocks = [DLinearBlock(d_model=d_model, kernel_size=5)]

    backbone = HybridBackbone(
        input_size=3,
        d_model=d_model,
        blocks=blocks,
        skip_input_proj=False,
    )

    x = torch.randn(2, 20, 3)
    out = backbone(x)
    assert out.shape == (2, d_model)
    assert isinstance(backbone.input_proj, torch.nn.Linear)


def test_channel_rejoin_mean_mode():
    """ChannelRejoin mean mode averages across channels."""
    rejoin = ChannelRejoin(num_channels=3, mode="mean")
    x = torch.randn(6, 10, 16)  # batch=2, channels=3
    out = rejoin(x)
    assert out.shape == (2, 10, 16)
