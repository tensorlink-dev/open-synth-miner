import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def test_hybrid_backbone_infers_output_dim_with_input_projection():
    cfg = OmegaConf.load("configs/model/hybrid_v2.yaml")
    backbone = instantiate(cfg.model.backbone)

    sample = torch.randn(2, 3, cfg.model.backbone.input_size)
    output = backbone(sample)

    assert backbone.output_dim == cfg.model.backbone.d_model
    assert output.shape == (2, cfg.model.backbone.d_model)
