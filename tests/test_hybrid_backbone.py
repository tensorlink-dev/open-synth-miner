import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.models.components.advanced_blocks import (
    ChannelRejoin,
    DLinearBlock,
    FlexiblePatchEmbed,
    LayerNormBlock,
)
from src.models.factory import HybridBackbone


def test_hybrid_backbone_infers_output_dim_with_input_projection():
    cfg = OmegaConf.load("configs/model/hybrid_v2.yaml")
    backbone = instantiate(cfg.model.backbone)

    sample = torch.randn(2, 3, cfg.model.backbone.input_size)
    output = backbone(sample)

    assert backbone.output_dim == cfg.model.backbone.d_model
    assert output.shape == (2, cfg.model.backbone.d_model)


def test_hybrid_backbone_automatic_layernorm_insertion():
    """Test that insert_layernorm=True inserts LayerNormBlock between blocks."""
    cfg = OmegaConf.load("configs/model/hybrid_with_layernorm.yaml")
    backbone = instantiate(cfg.model.backbone)

    # With 3 blocks, we should have: block1, layernorm, block2, layernorm, block3
    # Total 5 layers (3 original + 2 layernorms)
    assert len(backbone.layers) == 5, f"Expected 5 layers, got {len(backbone.layers)}"

    # Check that LayerNormBlock instances are inserted at even indices (1, 3)
    assert isinstance(backbone.layers[1], LayerNormBlock)
    assert isinstance(backbone.layers[3], LayerNormBlock)

    # Verify forward pass works
    sample = torch.randn(2, 10, cfg.model.backbone.input_size)
    output = backbone(sample)
    assert output.shape == (2, cfg.model.backbone.d_model)


def test_hybrid_backbone_manual_layernorm_insertion():
    """Test that manual LayerNormBlock insertion works as expected."""
    cfg = OmegaConf.load("configs/model/hybrid_manual_layernorm.yaml")
    backbone = instantiate(cfg.model.backbone)

    # With manual insertion: block1, layernorm, block2, layernorm, block3
    # Total 5 layers
    assert len(backbone.layers) == 5, f"Expected 5 layers, got {len(backbone.layers)}"

    # Check that LayerNormBlock instances are at indices 1 and 3
    assert isinstance(backbone.layers[1], LayerNormBlock)
    assert isinstance(backbone.layers[3], LayerNormBlock)

    # Verify forward pass works
    sample = torch.randn(2, 10, cfg.model.backbone.input_size)
    output = backbone(sample)
    assert output.shape == (2, cfg.model.backbone.d_model)


def test_hybrid_backbone_layernorm_block_shape_preservation():
    """Test that LayerNormBlock preserves (batch, seq, d_model) shape."""
    cfg = OmegaConf.load("configs/model/hybrid_with_layernorm.yaml")
    backbone = instantiate(cfg.model.backbone)

    batch_size = 4
    seq_len = 15
    sample = torch.randn(batch_size, seq_len, cfg.model.backbone.input_size)

    # Verify full sequence forward pass preserves shape through LayerNorms
    output_seq = backbone.forward_sequence(sample)
    assert output_seq.shape == (batch_size, seq_len, cfg.model.backbone.d_model)

    # Verify standard forward returns last step
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
    # Simulate (batch*channels, seq, d_model)
    x = torch.randn(6, 10, 16)  # batch=2, channels=3
    out = rejoin(x)
    assert out.shape == (2, 10, 16)
