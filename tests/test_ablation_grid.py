"""Tests for the ablation grid generator."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.models.factory import create_model
from src.models.registry import discover_components
from src.research.ablation_grid import (
    AblationGridSpec,
    ENGINEER_SPECS,
    REVIN_DLINEAR_COMBOS,
    generate_ablation_grid,
    describe_grid,
    _build_single_config,
)


@pytest.fixture(autouse=True)
def _discover():
    discover_components("src/models/components")


class TestGridGeneration:
    """Tests for generate_ablation_grid."""

    def test_default_grid_size(self):
        """Full grid: 2 engineers × (1+1+3+3) combos × 3 nheads = 48 configs.

        - none: 1 kernel_size sentinel × 3 nheads = 3
        - revin: 1 kernel_size sentinel × 3 nheads = 3
        - dlinear: 3 kernel_sizes × 3 nheads = 9
        - revin_dlinear: 3 kernel_sizes × 3 nheads = 9
        Per engineer: 3+3+9+9 = 24.  Two engineers: 48.
        """
        configs = generate_ablation_grid()
        assert len(configs) == 48

    def test_single_engineer(self):
        spec = AblationGridSpec(engineers=["zscore"])
        configs = generate_ablation_grid(spec)
        assert len(configs) == 24
        for name in configs:
            assert "eng=zscore" in name

    def test_single_nhead(self):
        spec = AblationGridSpec(nheads=[4])
        configs = generate_ablation_grid(spec)
        # 2 engineers × (1+1+3+3) = 16
        assert len(configs) == 16

    def test_no_dlinear_skips_kernel_axis(self):
        """When DLinear is excluded, kernel_sizes should not expand the grid."""
        spec = AblationGridSpec(revin_dlinear=["none", "revin"])
        configs = generate_ablation_grid(spec)
        # 2 engineers × 2 combos × 3 nheads = 12 (kernel_size collapsed)
        assert len(configs) == 12
        for name in configs:
            assert "__ks=" not in name

    def test_dlinear_only_expands_kernel(self):
        spec = AblationGridSpec(
            engineers=["zscore"],
            revin_dlinear=["dlinear"],
            nheads=[4],
        )
        configs = generate_ablation_grid(spec)
        assert len(configs) == 3  # 3 kernel sizes
        for name in configs:
            assert "__ks=" in name

    def test_invalid_nhead_skipped(self):
        """d_model=32 is not divisible by 5, so nhead=5 should be pruned."""
        spec = AblationGridSpec(
            engineers=["zscore"],
            revin_dlinear=["none"],
            nheads=[4, 5],
        )
        configs = generate_ablation_grid(spec)
        assert len(configs) == 1
        assert "nh=4" in list(configs.keys())[0]

    def test_training_overrides_applied(self):
        spec = AblationGridSpec(
            engineers=["zscore"],
            revin_dlinear=["none"],
            nheads=[4],
        )
        configs = generate_ablation_grid(spec, training_overrides={"epochs": 20, "lr": 0.01})
        cfg = next(iter(configs.values()))
        assert cfg.training.epochs == 20
        assert cfg.training.lr == 0.01


class TestConfigValidity:
    """Verify that generated configs produce valid models."""

    @pytest.mark.parametrize("engineer", list(ENGINEER_SPECS.keys()))
    def test_model_instantiation_per_engineer(self, engineer):
        spec = AblationGridSpec(
            engineers=[engineer],
            revin_dlinear=["none"],
            nheads=[4],
        )
        configs = generate_ablation_grid(spec)
        assert len(configs) == 1
        cfg = next(iter(configs.values()))
        model = create_model(cfg)
        assert model is not None

    @pytest.mark.parametrize("combo", list(REVIN_DLINEAR_COMBOS.keys()))
    def test_model_instantiation_per_revin_dlinear(self, combo):
        spec = AblationGridSpec(
            engineers=["zscore"],
            revin_dlinear=[combo],
            nheads=[4],
            kernel_sizes=[25],
        )
        configs = generate_ablation_grid(spec)
        assert len(configs) >= 1
        for name, cfg in configs.items():
            model = create_model(cfg)
            feature_dim = cfg.training.feature_dim
            x = torch.randn(2, 32, feature_dim)
            price = torch.ones(2)
            with torch.no_grad():
                paths, mu, sigma = model(x, price, horizon=4, n_paths=5)
            assert paths.shape == (2, 5, 4)

    @pytest.mark.parametrize("nhead", [2, 4, 8])
    def test_model_forward_per_nhead(self, nhead):
        spec = AblationGridSpec(
            engineers=["zscore"],
            revin_dlinear=["none"],
            nheads=[nhead],
        )
        configs = generate_ablation_grid(spec)
        cfg = next(iter(configs.values()))
        model = create_model(cfg)
        x = torch.randn(2, 32, 3)
        price = torch.ones(2)
        with torch.no_grad():
            paths, mu, sigma = model(x, price, horizon=4, n_paths=5)
        assert paths.shape == (2, 5, 4)

    @pytest.mark.parametrize("ks", [15, 25, 51])
    def test_model_forward_per_kernel_size(self, ks):
        spec = AblationGridSpec(
            engineers=["zscore"],
            revin_dlinear=["dlinear"],
            nheads=[4],
            kernel_sizes=[ks],
        )
        configs = generate_ablation_grid(spec)
        cfg = next(iter(configs.values()))
        model = create_model(cfg)
        x = torch.randn(2, 32, 3)
        price = torch.ones(2)
        with torch.no_grad():
            paths, mu, sigma = model(x, price, horizon=4, n_paths=5)
        assert paths.shape == (2, 5, 4)

    def test_full_revin_dlinear_forward(self):
        """Full combo with RevIN + DLinear should forward correctly."""
        spec = AblationGridSpec(
            engineers=["wavelet"],
            revin_dlinear=["revin_dlinear"],
            nheads=[4],
            kernel_sizes=[25],
        )
        configs = generate_ablation_grid(spec)
        cfg = next(iter(configs.values()))
        model = create_model(cfg)
        x = torch.randn(2, 32, 5)  # wavelet: 5 features
        price = torch.ones(2)
        with torch.no_grad():
            paths, mu, sigma = model(x, price, horizon=8, n_paths=10)
        assert paths.shape == (2, 10, 8)


class TestConfigContent:
    """Verify config structure and values."""

    def test_feature_dim_matches_engineer(self):
        for eng_name, eng_spec in ENGINEER_SPECS.items():
            raw = _build_single_config(
                engineer_name=eng_name,
                revin_dlinear_name="none",
                kernel_size=25,
                nhead=4,
                d_model=32,
                head_target="src.models.heads.GBMHead",
            )
            assert raw["model"]["backbone"]["input_size"] == eng_spec["feature_dim"]
            assert raw["training"]["feature_dim"] == eng_spec["feature_dim"]

    def test_blocks_include_revin_when_specified(self):
        raw = _build_single_config(
            engineer_name="zscore",
            revin_dlinear_name="revin",
            kernel_size=25,
            nhead=4,
            d_model=32,
            head_target="src.models.heads.GBMHead",
        )
        targets = [b["_target_"] for b in raw["model"]["backbone"]["blocks"]]
        assert any("RevIN" in t for t in targets)
        assert not any("DLinear" in t for t in targets)

    def test_blocks_include_dlinear_with_kernel(self):
        raw = _build_single_config(
            engineer_name="zscore",
            revin_dlinear_name="dlinear",
            kernel_size=51,
            nhead=4,
            d_model=32,
            head_target="src.models.heads.GBMHead",
        )
        blocks = raw["model"]["backbone"]["blocks"]
        dlinear_blocks = [b for b in blocks if "DLinear" in b["_target_"]]
        assert len(dlinear_blocks) == 1
        assert dlinear_blocks[0]["kernel_size"] == 51

    def test_nhead_propagated_to_transformer(self):
        raw = _build_single_config(
            engineer_name="zscore",
            revin_dlinear_name="none",
            kernel_size=25,
            nhead=8,
            d_model=32,
            head_target="src.models.heads.GBMHead",
        )
        blocks = raw["model"]["backbone"]["blocks"]
        transformer_blocks = [b for b in blocks if "Transformer" in b["_target_"]]
        assert len(transformer_blocks) == 1
        assert transformer_blocks[0]["nhead"] == 8


class TestDescribeGrid:
    def test_describe_output(self):
        configs = generate_ablation_grid(AblationGridSpec(
            engineers=["zscore"],
            revin_dlinear=["none"],
            nheads=[4],
        ))
        output = describe_grid(configs)
        assert "1 configurations" in output
        assert "eng=zscore" in output
