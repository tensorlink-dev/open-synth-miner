import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.models.components.advanced_blocks import LayerNormBlock


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
