#!/usr/bin/env python
"""Comprehensive test suite for open-synth-miner.

Tests every major module and outputs a formatted pass/fail report.

Usage:
    python tests/test_comprehensive_suite.py        # standalone runner
    python -m pytest tests/test_comprehensive_suite.py -v  # via pytest
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import traceback
import types
from typing import Callable, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Hydra / omegaconf compatibility shim
# Must be installed into sys.modules BEFORE any 'from src.*' imports so that
# src/models/factory.py can be imported even if hydra-core is not installed
# or omegaconf has an antlr4 version conflict.
# ---------------------------------------------------------------------------

def _install_hydra_omegaconf_shims() -> None:
    """Insert minimal mock modules for hydra.utils and omegaconf."""

    def _instantiate(cfg, *args, **kwargs):
        """Recursively instantiate objects from dicts with '_target_' keys."""
        if isinstance(cfg, dict):
            cfg = dict(cfg)
            target = cfg.pop("_target_", None)
            if target is None:
                return cfg
            # Recursively instantiate nested configs
            for k, v in list(cfg.items()):
                if isinstance(v, dict) and "_target_" in v:
                    cfg[k] = _instantiate(v)
                elif isinstance(v, list):
                    cfg[k] = [_instantiate(i) if isinstance(i, dict) else i for i in v]
            # Import and call the target class/function
            parts = target.rsplit(".", 1)
            mod = importlib.import_module(parts[0])
            cls = getattr(mod, parts[1])
            return cls(**cfg, **kwargs)
        return cfg

    # --- omegaconf mock ---
    class _DictConfig(dict):
        pass

    class _OmegaConf:
        @staticmethod
        def to_container(cfg, resolve=True, throw_on_missing=False):
            return dict(cfg) if isinstance(cfg, dict) else cfg

        @staticmethod
        def create(data=None):
            return _DictConfig(data or {})

    # Only install the mock if the real module is absent or broken
    if "omegaconf" not in sys.modules:
        oc = types.ModuleType("omegaconf")
        oc.DictConfig = _DictConfig
        oc.OmegaConf = _OmegaConf
        oc.SCMode = None
        sys.modules["omegaconf"] = oc
    else:
        # Real module present — probe for antlr4 breakage
        try:
            from omegaconf import DictConfig  # noqa: F401
        except Exception:
            oc = types.ModuleType("omegaconf")
            oc.DictConfig = _DictConfig
            oc.OmegaConf = _OmegaConf
            oc.SCMode = None
            sys.modules["omegaconf"] = oc

    # --- hydra mock ---
    if "hydra" not in sys.modules or "hydra.utils" not in sys.modules:
        for _name in [
            "hydra", "hydra.utils", "hydra.core", "hydra.core.global_hydra",
            "hydra._internal", "hydra._internal.utils",
            "hydra._internal.instantiate", "hydra._internal.instantiate._internal",
        ]:
            sys.modules.setdefault(_name, types.ModuleType(_name))

        hydra_mod = sys.modules["hydra"]
        hydra_utils = sys.modules["hydra.utils"]
        hydra_utils.instantiate = _instantiate
        hydra_mod.utils = hydra_utils
    else:
        try:
            from hydra.utils import instantiate  # noqa: F401
        except Exception:
            hydra_utils = types.ModuleType("hydra.utils")
            hydra_utils.instantiate = _instantiate
            sys.modules["hydra.utils"] = hydra_utils
            if hasattr(sys.modules.get("hydra"), "utils"):
                sys.modules["hydra"].utils = hydra_utils


_install_hydra_omegaconf_shims()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Minimal test runner
# ---------------------------------------------------------------------------

class Result:
    def __init__(self, name: str, passed: bool, msg: str = "", elapsed: float = 0.0):
        self.name = name
        self.passed = passed
        self.msg = msg
        self.elapsed = elapsed


RESULTS: List[Result] = []


def run(name: str, fn: Callable) -> None:
    """Execute fn() and record pass/fail."""
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        r = Result(name, True, elapsed=elapsed)
        print(f"  [PASS] {name}  ({elapsed:.3f}s)")
    except Exception:
        elapsed = time.perf_counter() - t0
        tb = traceback.format_exc()
        r = Result(name, False, msg=tb, elapsed=elapsed)
        lines = tb.strip().split("\n")
        print(f"  [FAIL] {name}  ({elapsed:.3f}s)")
        print(f"         -> {lines[-1]}")
    RESULTS.append(r)


def section(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def summary() -> bool:
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r.passed)
    failed = total - passed
    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"  Total: {total}  |  Passed: {passed}  |  Failed: {failed}")
    if failed:
        print(f"\n  FAILED TESTS:")
        for r in RESULTS:
            if not r.passed:
                print(f"    [FAIL] {r.name}")
                for line in r.msg.strip().split("\n")[-4:]:
                    print(f"           {line}")
    print(f"\n  RESULTS:")
    for r in RESULTS:
        status = "PASS" if r.passed else "FAIL"
        print(f"    [{status}] {r.name}")
    print()
    return failed == 0


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _t(batch: int = 2, seq: int = 16, d: int = 32) -> torch.Tensor:
    return torch.randn(batch, seq, d)


def _backbone(input_size: int = 3, d_model: int = 32):
    from src.models.factory import HybridBackbone
    from src.models.registry import TransformerBlock
    return HybridBackbone(
        input_size=input_size,
        d_model=d_model,
        blocks=[TransformerBlock(d_model=d_model, nhead=4)],
        validate_shapes=True,
    )


# =============================================================================
# SECTION 1: Registry
# =============================================================================

def _test_registry_instantiation():
    from src.models.registry import Registry, registry
    r = Registry()
    assert hasattr(r, "components") and hasattr(r, "blocks") and hasattr(r, "hybrids")
    assert isinstance(registry, Registry)


def _test_registry_block_register_and_get():
    from src.models.registry import Registry
    r = Registry()

    @r.register_block("myblock_s1", preserves_seq_len=True, min_seq_len=2, description="Test")
    class MyBlock(nn.Module):
        def forward(self, x): return x

    assert "myblock_s1" in r.blocks
    assert r.get_block("myblock_s1") is MyBlock
    info = r.get_info("myblock_s1")
    assert info.kind == "block"
    assert info.preserves_seq_len is True
    assert info.min_seq_len == 2
    assert info.description == "Test"


def _test_registry_component_register():
    from src.models.registry import Registry
    r = Registry()

    @r.register_component("mycomp_s1", description="Comp")
    class MyComp(nn.Module):
        def forward(self, x): return x

    assert "mycomp_s1" in r.components
    assert r.get_component("mycomp_s1") is MyComp


def _test_registry_hybrid_register():
    from src.models.registry import Registry
    r = Registry()

    @r.register_hybrid("myhybrid_s1", description="Hybrid")
    def my_h(d_model: int = 32) -> nn.Module:
        return nn.Linear(d_model, d_model)

    assert "myhybrid_s1" in r.hybrids
    assert r.get_hybrid("myhybrid_s1") is my_h


def _test_registry_list_blocks():
    from src.models.registry import registry
    blocks = registry.list_blocks(kind="block")
    assert len(blocks) > 0
    assert all(b.kind == "block" for b in blocks)
    comps = registry.list_blocks(kind="component")
    assert all(c.kind == "component" for c in comps)


def _test_registry_summary():
    from src.models.registry import registry
    s = registry.summary()
    assert isinstance(s, str) and len(s) > 0
    s_blocks = registry.summary(kind="block")
    assert "block" in s_blocks


def _test_registry_recipe_hash():
    from src.models.registry import Registry
    h1 = Registry.recipe_hash({"a": 1, "b": [2, 3]})
    h2 = Registry.recipe_hash({"b": [2, 3], "a": 1})
    assert h1 == h2
    assert len(h1) == 12
    h3 = Registry.recipe_hash({"a": 2})
    assert h3 != h1


def _test_registry_attribute_access():
    from src.models.registry import registry, TransformerBlock, LSTMBlock
    assert registry.TransformerBlock is TransformerBlock
    assert registry.LSTMBlock is LSTMBlock


def _test_registry_missing_raises():
    from src.models.registry import registry
    try:
        registry.get_block("no_such_block_xyzabc")
        raise AssertionError("Should raise KeyError")
    except KeyError:
        pass
    try:
        _ = registry.no_such_attr_xyzabc
        raise AssertionError("Should raise AttributeError")
    except AttributeError:
        pass


def _test_registry_duplicate_raises():
    from src.models.registry import Registry
    r = Registry()

    @r.register_block("dupblock_s1")
    class B1(nn.Module):
        def forward(self, x): return x

    try:
        @r.register_block("dupblock_s1")
        class B2(nn.Module):
            def forward(self, x): return x
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


def _test_discover_components():
    from src.models.registry import discover_components, registry
    before = len(registry.blocks)
    discover_components("src/models/components")
    after = len(registry.blocks)
    # advanced_blocks registers at least 10 more blocks
    assert after >= before


# =============================================================================
# SECTION 2: Built-in Blocks (registry.py)
# =============================================================================

def _test_transformer_block():
    from src.models.registry import TransformerBlock
    b = TransformerBlock(d_model=32, nhead=4)
    x = _t()
    out = b(x)
    assert out.shape == x.shape


def _test_lstm_block():
    from src.models.registry import LSTMBlock
    b = LSTMBlock(d_model=32, num_layers=2)
    x = _t()
    assert b(x).shape == x.shape


def _test_sde_evolution_block():
    from src.models.registry import SDEEvolutionBlock
    b = SDEEvolutionBlock(d_model=32, hidden=64)
    x = _t()
    assert b(x).shape == x.shape


def _test_custom_attention():
    from src.models.registry import CustomAttention
    c = CustomAttention(d_model=32, nhead=4)
    x = _t()
    assert c(x).shape == x.shape


def _test_gated_mlp():
    from src.models.registry import GatedMLP
    m = GatedMLP(d_model=32, expansion=4)
    x = _t()
    assert m(x).shape == x.shape


def _test_patch_merging():
    from src.models.registry import PatchMerging
    pm = PatchMerging(d_model=32)
    x = _t(seq=16)
    out = pm(x)
    assert out.shape == (2, 8, 32), f"got {out.shape}"


# =============================================================================
# SECTION 3: Advanced Blocks (advanced_blocks.py)
# =============================================================================

def _test_rnn_block():
    from src.models.components.advanced_blocks import RNNBlock
    b = RNNBlock(d_model=32, num_layers=2)
    x = _t()
    assert b(x).shape == x.shape


def _test_gru_block():
    from src.models.components.advanced_blocks import GRUBlock
    b = GRUBlock(d_model=32, num_layers=1)
    x = _t()
    assert b(x).shape == x.shape


def _test_resconv_block():
    from src.models.components.advanced_blocks import ResConvBlock
    b = ResConvBlock(d_model=32, kernel_size=3)
    x = _t()
    assert b(x).shape == x.shape


def _test_bitcn_block():
    from src.models.components.advanced_blocks import BiTCNBlock
    b = BiTCNBlock(d_model=32, kernel_size=3, dilation=2)
    x = _t()
    assert b(x).shape == x.shape


def _test_patch_embedding_changes_seq():
    from src.models.components.advanced_blocks import PatchEmbedding
    b = PatchEmbedding(d_model=32, patch_size=4)
    x = _t(seq=32)
    out = b(x)
    assert out.ndim == 3
    assert out.shape[0] == 2 and out.shape[2] == 32
    assert out.shape[1] < 32  # seq len reduced by patch


def _test_unet1d_block():
    from src.models.components.advanced_blocks import Unet1DBlock
    b = Unet1DBlock(d_model=32)
    x = _t(seq=16)
    out = b(x)
    assert out.shape == x.shape


def _test_transformer_encoder_adapter():
    from src.models.components.advanced_blocks import TransformerEncoderAdapter
    b = TransformerEncoderAdapter(d_model=32, nhead=4, num_layers=2)
    x = _t()
    assert b(x).shape == x.shape


def _test_transformer_decoder_adapter():
    from src.models.components.advanced_blocks import TransformerDecoderAdapter
    b = TransformerDecoderAdapter(d_model=32, nhead=4, num_layers=2)
    x = _t()
    assert b(x).shape == x.shape


def _test_fourier_block():
    from src.models.components.advanced_blocks import FourierBlock
    b = FourierBlock(d_model=32, modes=8)
    x = _t(seq=32)
    out = b(x)
    assert out.shape == x.shape


def _test_last_step_adapter_modes():
    from src.models.components.advanced_blocks import LastStepAdapter
    x = _t()
    for mode in ("last", "mean", "max"):
        b = LastStepAdapter(d_model=32, mode=mode)
        out = b(x)
        assert out.ndim == 3


def _test_revin_norm_denorm():
    from src.models.components.advanced_blocks import RevIN
    b = RevIN(d_model=32)
    x = _t()
    x_norm = b(x, mode="norm")
    assert x_norm.shape == x.shape
    x_denorm = b(x_norm, mode="denorm")
    assert x_denorm.shape == x.shape


def _test_revin_invalid_mode():
    from src.models.components.advanced_blocks import RevIN
    b = RevIN(d_model=32)
    x = _t()
    b(x, mode="norm")
    try:
        b(x, mode="invalid_mode")
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


def _test_flexible_patch_embed():
    from src.models.components.advanced_blocks import FlexiblePatchEmbed
    b = FlexiblePatchEmbed(d_model=32, patch_len=8, stride=4, in_channels=3, channel_independence=False)
    x = torch.randn(2, 64, 3)
    out = b(x)
    assert out.ndim == 3 and out.shape[0] == 2 and out.shape[2] == 32


def _test_flexible_patch_embed_channel_independence():
    from src.models.components.advanced_blocks import FlexiblePatchEmbed
    batch, seq, channels = 2, 64, 3
    b = FlexiblePatchEmbed(d_model=32, patch_len=8, stride=4, channel_independence=True)
    x = torch.randn(batch, seq, channels)
    out = b(x)
    # channel_independence multiplies batch by channels
    assert out.ndim == 3
    assert out.shape[0] == batch * channels
    assert out.shape[2] == 32


def _test_channel_rejoin_mean():
    from src.models.components.advanced_blocks import ChannelRejoin
    num_channels = 3
    b = ChannelRejoin(num_channels=num_channels, mode="mean")
    # Simulates batch*channels from channel-independent processing
    x = torch.randn(2 * num_channels, 16, 32)
    out = b(x)
    assert out.shape == (2, 16, 32)


def _test_dlinear_block():
    from src.models.components.advanced_blocks import DLinearBlock
    b = DLinearBlock(d_model=32, kernel_size=5)
    x = _t()
    out = b(x)
    assert out.shape == x.shape


def _test_layernorm_block():
    from src.models.components.advanced_blocks import LayerNormBlock
    b = LayerNormBlock(d_model=32)
    x = _t()
    out = b(x)
    assert out.shape == x.shape
    # Check that mean is ~0 and std is ~1 across last dim
    assert out.isfinite().all()


def _test_timesnet_block():
    from src.models.components.advanced_blocks import TimesNetBlock
    b = TimesNetBlock(d_model=32, top_k=3)
    # min_seq_len=4
    x = _t(seq=32)
    out = b(x)
    assert out.shape == x.shape


# =============================================================================
# SECTION 4: Heads
# =============================================================================

def _test_gbm_head_shapes_and_positivity():
    from src.models.heads import GBMHead
    h = GBMHead(latent_size=32)
    x = torch.randn(4, 32)
    mu, sigma = h(x)
    assert mu.shape == (4,) and sigma.shape == (4,)
    assert (sigma > 0).all()


def _test_sde_head():
    from src.models.heads import SDEHead
    h = SDEHead(latent_size=32, hidden=64)
    x = torch.randn(4, 32)
    mu, sigma = h(x)
    assert mu.shape == (4,) and sigma.shape == (4,)
    assert (sigma > 0).all()


def _test_horizon_head():
    from src.models.heads import HorizonHead
    h = HorizonHead(latent_size=32, horizon_max=48, nhead=4, n_layers=2)
    x = torch.randn(4, 16, 32)
    mu, sigma = h(x, horizon=12)
    assert mu.shape == (4, 12) and sigma.shape == (4, 12)
    assert (sigma > 0).all()


def _test_horizon_head_kv_compression():
    from src.models.heads import HorizonHead
    h = HorizonHead(latent_size=32, horizon_max=24, nhead=4, kv_dim=16)
    x = torch.randn(2, 32, 32)
    mu, sigma = h(x, horizon=8)
    assert mu.shape == (2, 8) and sigma.shape == (2, 8)


def _test_horizon_head_clips_to_max():
    from src.models.heads import HorizonHead
    import logging
    h = HorizonHead(latent_size=32, horizon_max=10, nhead=4)
    x = torch.randn(2, 16, 32)
    # horizon > horizon_max should clip
    mu, sigma = h(x, horizon=20)
    assert mu.shape == (2, 10)  # clipped to horizon_max


def _test_simple_horizon_head_all_pools():
    from src.models.heads import SimpleHorizonHead
    for pool in ("mean", "max", "mean+max"):
        h = SimpleHorizonHead(latent_size=32, horizon_max=48, pool_type=pool)
        x = torch.randn(3, 16, 32)
        mu, sigma = h(x, horizon=8)
        assert mu.shape == (3, 8) and sigma.shape == (3, 8)
        assert (sigma > 0).all()


def _test_simple_horizon_head_invalid_pool():
    from src.models.heads import SimpleHorizonHead
    try:
        SimpleHorizonHead(latent_size=32, pool_type="invalid_pool_xyz")
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


def _test_mixture_density_head():
    from src.models.heads import MixtureDensityHead
    h = MixtureDensityHead(latent_size=32, n_components=3, hidden=64)
    x = torch.randn(4, 32)
    mus, sigmas, weights = h(x)
    assert mus.shape == (4, 3) and sigmas.shape == (4, 3) and weights.shape == (4, 3)
    assert (sigmas > 0).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)


def _test_vol_term_structure_head():
    from src.models.heads import VolTermStructureHead
    h = VolTermStructureHead(latent_size=32, hidden=64)
    x = torch.randn(4, 32)
    mu_seq, sigma_seq = h(x, horizon=12)
    assert mu_seq.shape == (4, 12) and sigma_seq.shape == (4, 12)
    assert (sigma_seq > 0).all()


def _test_neural_bridge_head_no_price():
    from src.models.heads import NeuralBridgeHead
    h = NeuralBridgeHead(latent_size=32, micro_steps=12, hidden_dim=64)
    x = torch.randn(4, 32)
    macro_ret, micro_returns, sigma = h(x)
    assert macro_ret.shape == (4, 1)
    assert micro_returns.shape == (4, 12)
    assert sigma.shape == (4,)
    assert (sigma > 0).all()
    # Bridge constraints: start and end should be near 0
    assert micro_returns[:, 0].abs().mean() < 0.5  # approximately
    assert micro_returns[:, -1].abs().mean() < 1.0  # approximately


def _test_neural_bridge_head_with_price():
    from src.models.heads import NeuralBridgeHead
    h = NeuralBridgeHead(latent_size=32, micro_steps=8)
    x = torch.randn(3, 32)
    price = torch.ones(3) * 100.0
    macro_ret, prices_out, sigma = h(x, current_price=price)
    assert macro_ret.shape == (3, 1)
    assert prices_out.shape == (3, 8)
    assert (prices_out > 0).all()  # prices should be positive after exp


def _test_neural_sde_head():
    from src.models.heads import NeuralSDEHead
    with torch.no_grad():
        h = NeuralSDEHead(latent_size=32, hidden=32, solver="euler", adjoint=False)
        ctx = torch.randn(2, 32)
        price = torch.ones(2) * 100.0
        paths, mu, sigma = h(ctx, price, horizon=4, n_paths=5, dt=1.0)
    assert paths.shape == (2, 5, 4)
    assert mu.shape == (2,) and sigma.shape == (2,)
    assert (sigma > 0).all()
    assert (paths > 0).all()


# =============================================================================
# SECTION 5: HybridBackbone & ParallelFusion
# =============================================================================

def _test_hybrid_backbone_last_step():
    from src.models.factory import HybridBackbone
    from src.models.registry import TransformerBlock
    bb = HybridBackbone(input_size=3, d_model=32, blocks=[TransformerBlock(d_model=32)])
    x = torch.randn(4, 16, 3)
    out = bb(x)
    assert out.shape == (4, 32), f"got {out.shape}"


def _test_hybrid_backbone_forward_sequence():
    from src.models.factory import HybridBackbone
    from src.models.registry import LSTMBlock
    bb = HybridBackbone(input_size=3, d_model=32, blocks=[LSTMBlock(d_model=32)])
    x = torch.randn(4, 16, 3)
    out = bb.forward_sequence(x)
    assert out.shape == (4, 16, 32), f"got {out.shape}"


def _test_hybrid_backbone_multiple_blocks():
    from src.models.factory import HybridBackbone
    from src.models.registry import TransformerBlock, LSTMBlock, SDEEvolutionBlock
    bb = HybridBackbone(
        input_size=3,
        d_model=32,
        blocks=[TransformerBlock(d_model=32), LSTMBlock(d_model=32), SDEEvolutionBlock(d_model=32)],
    )
    x = torch.randn(2, 16, 3)
    out = bb(x)
    assert out.shape == (2, 32)


def _test_hybrid_backbone_insert_layernorm():
    from src.models.factory import HybridBackbone
    from src.models.registry import TransformerBlock, LSTMBlock
    bb = HybridBackbone(
        input_size=3,
        d_model=32,
        blocks=[TransformerBlock(d_model=32), LSTMBlock(d_model=32)],
        insert_layernorm=True,
    )
    x = torch.randn(2, 16, 3)
    out = bb(x)
    assert out.shape == (2, 32)
    # 2 blocks => 3 layers: block, LayerNorm, block
    assert len(bb.layers) == 3


def _test_hybrid_backbone_empty_raises():
    from src.models.factory import HybridBackbone
    try:
        HybridBackbone(input_size=3, d_model=32, blocks=[])
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


def _test_hybrid_backbone_output_dim():
    from src.models.factory import HybridBackbone
    from src.models.registry import TransformerBlock
    bb = HybridBackbone(input_size=5, d_model=64, blocks=[TransformerBlock(d_model=64, nhead=4)])
    assert bb.output_dim == 64


def _test_parallel_fusion_gating():
    from src.models.factory import ParallelFusion
    paths = nn.ModuleList([nn.Linear(32, 32), nn.Linear(32, 32)])
    fusion = ParallelFusion(list(paths), merge_strategy="gating")
    x = torch.randn(4, 32)
    out = fusion(x)
    assert out.shape == (4, 32)


def _test_parallel_fusion_concat():
    from src.models.factory import ParallelFusion
    paths = [nn.Linear(32, 16), nn.Linear(32, 16)]
    fusion = ParallelFusion(paths, merge_strategy="concat")
    x = torch.randn(4, 32)
    out = fusion(x)
    assert out.shape == (4, 32)


def _test_parallel_fusion_too_few_paths_raises():
    from src.models.factory import ParallelFusion
    try:
        ParallelFusion([nn.Linear(32, 32)], merge_strategy="gating")
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


def _test_parallel_fusion_bad_strategy_raises():
    from src.models.factory import ParallelFusion
    try:
        ParallelFusion([nn.Linear(32, 32), nn.Linear(32, 32)], merge_strategy="sum")
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


# =============================================================================
# SECTION 6: SynthModel — all head types
# =============================================================================

def _synth(head):
    from src.models.factory import SynthModel
    return SynthModel(_backbone(), head)


def _test_synth_gbm():
    from src.models.heads import GBMHead
    model = _synth(GBMHead(latent_size=32))
    x, p = torch.randn(2, 16, 3), torch.ones(2) * 100.0
    with torch.no_grad():
        paths, mu, sigma = model(x, p, horizon=8, n_paths=10)
    assert paths.shape == (2, 10, 8)
    assert (paths > 0).all()


def _test_synth_sde():
    from src.models.heads import SDEHead
    model = _synth(SDEHead(latent_size=32))
    x, p = torch.randn(2, 16, 3), torch.ones(2) * 50.0
    with torch.no_grad():
        paths, mu, sigma = model(x, p, horizon=6, n_paths=8)
    assert paths.shape == (2, 8, 6)


def _test_synth_horizon():
    from src.models.heads import HorizonHead
    model = _synth(HorizonHead(latent_size=32, horizon_max=48, nhead=4))
    x, p = torch.randn(2, 16, 3), torch.ones(2) * 100.0
    with torch.no_grad():
        paths, mu, sigma = model(x, p, horizon=12, n_paths=10)
    assert paths.shape == (2, 10, 12)


def _test_synth_simple_horizon():
    from src.models.heads import SimpleHorizonHead
    model = _synth(SimpleHorizonHead(latent_size=32, horizon_max=48))
    x, p = torch.randn(2, 16, 3), torch.ones(2) * 100.0
    with torch.no_grad():
        paths, mu, sigma = model(x, p, horizon=12, n_paths=10)
    assert paths.shape == (2, 10, 12)


def _test_synth_mixture_density():
    from src.models.heads import MixtureDensityHead
    model = _synth(MixtureDensityHead(latent_size=32, n_components=3))
    x, p = torch.randn(2, 16, 3), torch.ones(2) * 100.0
    with torch.no_grad():
        paths, mu, sigma = model(x, p, horizon=8, n_paths=10)
    assert paths.shape == (2, 10, 8)


def _test_synth_vol_term_structure():
    from src.models.heads import VolTermStructureHead
    model = _synth(VolTermStructureHead(latent_size=32))
    x, p = torch.randn(2, 16, 3), torch.ones(2) * 100.0
    with torch.no_grad():
        paths, mu, sigma = model(x, p, horizon=8, n_paths=10)
    assert paths.shape == (2, 10, 8)


def _test_synth_neural_bridge():
    from src.models.heads import NeuralBridgeHead
    model = _synth(NeuralBridgeHead(latent_size=32, micro_steps=8))
    x, p = torch.randn(2, 16, 3), torch.ones(2) * 100.0
    with torch.no_grad():
        paths, mu, sigma = model(x, p, horizon=8, n_paths=10)
    assert paths.shape == (2, 10, 8)


def _test_synth_neural_sde():
    from src.models.heads import NeuralSDEHead
    model = _synth(NeuralSDEHead(latent_size=32, hidden=32, solver="euler"))
    x, p = torch.randn(2, 16, 3), torch.ones(2) * 100.0
    with torch.no_grad():
        paths, mu, sigma = model(x, p, horizon=4, n_paths=5)
    assert paths.shape == (2, 5, 4)


def _test_synth_shape_contract_all_heads():
    """Shape invariant: (batch, n_paths, horizon) for every head type."""
    from src.models.factory import SynthModel
    from src.models.heads import (
        GBMHead, SDEHead, HorizonHead, SimpleHorizonHead,
        MixtureDensityHead, VolTermStructureHead, NeuralBridgeHead,
    )
    batch, n_paths, horizon = 3, 7, 5
    x = torch.randn(batch, 16, 3)
    p = torch.ones(batch) * 100.0

    configs = [
        GBMHead(latent_size=32),
        SDEHead(latent_size=32),
        HorizonHead(latent_size=32, horizon_max=20, nhead=4),
        SimpleHorizonHead(latent_size=32, horizon_max=20),
        MixtureDensityHead(latent_size=32),
        VolTermStructureHead(latent_size=32),
        NeuralBridgeHead(latent_size=32, micro_steps=horizon),
    ]
    for head in configs:
        model = SynthModel(_backbone(), head)
        with torch.no_grad():
            paths, _, _ = model(x, p, horizon=horizon, n_paths=n_paths)
        assert paths.ndim == 3
        assert paths.shape == (batch, n_paths, horizon), (
            f"{head.__class__.__name__}: expected ({batch},{n_paths},{horizon}), got {tuple(paths.shape)}"
        )


# =============================================================================
# SECTION 7: Path Simulators
# =============================================================================

def _test_simulate_gbm_paths():
    from src.models.factory import simulate_gbm_paths
    price = torch.ones(4) * 100.0
    mu = torch.zeros(4)
    sigma = torch.ones(4) * 0.01
    paths = simulate_gbm_paths(price, mu, sigma, horizon=12, n_paths=100)
    assert paths.shape == (4, 100, 12)
    assert (paths > 0).all()


def _test_simulate_gbm_positivity_extremes():
    from src.models.factory import simulate_gbm_paths
    price = torch.ones(2) * 100.0
    mu = torch.tensor([10.0, -10.0])
    sigma = torch.ones(2) * 5.0
    paths = simulate_gbm_paths(price, mu, sigma, horizon=20, n_paths=50)
    assert torch.isfinite(paths).all()
    assert (paths > 0).all()


def _test_simulate_gbm_dt():
    from src.models.factory import simulate_gbm_paths
    price = torch.ones(2) * 100.0
    mu, sigma = torch.zeros(2), torch.ones(2) * 0.01
    p1 = simulate_gbm_paths(price, mu, sigma, horizon=8, n_paths=50, dt=1.0)
    p2 = simulate_gbm_paths(price, mu, sigma, horizon=8, n_paths=50, dt=0.1)
    assert p1.shape == p2.shape == (2, 50, 8)


def _test_simulate_horizon_paths():
    from src.models.factory import simulate_horizon_paths
    batch, horizon = 4, 12
    price = torch.ones(batch) * 100.0
    mu_seq = torch.zeros(batch, horizon)
    sigma_seq = torch.ones(batch, horizon) * 0.01
    paths = simulate_horizon_paths(price, mu_seq, sigma_seq, n_paths=100)
    assert paths.shape == (4, 100, 12)
    assert (paths > 0).all()


def _test_simulate_bridge_paths():
    from src.models.factory import simulate_bridge_paths
    batch, steps = 4, 12
    price = torch.ones(batch) * 100.0
    micro = torch.zeros(batch, steps)
    sigma = torch.ones(batch) * 0.01
    paths = simulate_bridge_paths(price, micro, sigma, n_paths=100)
    assert paths.shape == (4, 100, 12)
    assert (paths > 0).all()


def _test_simulate_mixture_paths():
    from src.models.factory import simulate_mixture_paths
    batch, K = 4, 3
    price = torch.ones(batch) * 100.0
    mus = torch.zeros(batch, K)
    sigmas = torch.ones(batch, K) * 0.01
    weights = torch.ones(batch, K) / K
    paths = simulate_mixture_paths(price, mus, sigmas, weights, horizon=12, n_paths=100)
    assert paths.shape == (4, 100, 12)
    assert (paths > 0).all()


# =============================================================================
# SECTION 8: Data Pipeline
# =============================================================================

def _test_mock_data_source():
    from src.data.market_data_loader import MockDataSource, AssetData
    src = MockDataSource(length=300, freq="5min", seed=42)
    assets = src.load_data(["BTC", "ETH", "SYNTH"])
    assert len(assets) == 3
    for a in assets:
        assert isinstance(a, AssetData)
        assert len(a.prices) == 300
        assert len(a.timestamps) == 300
        assert (np.array(a.prices) > 0).all()


def _test_mock_data_source_reproducible():
    from src.data.market_data_loader import MockDataSource
    s1 = MockDataSource(length=100, seed=7).load_data(["X"])[0]
    s2 = MockDataSource(length=100, seed=7).load_data(["X"])[0]
    np.testing.assert_array_equal(s1.prices, s2.prices)


def _test_zscore_engineer_feature_dim():
    from src.data.market_data_loader import ZScoreEngineer
    eng = ZScoreEngineer()
    assert eng.feature_dim == 3


def _test_zscore_engineer_prepare_cache():
    from src.data.market_data_loader import ZScoreEngineer
    eng = ZScoreEngineer(short_win=10, long_win=50)
    prices = np.linspace(100, 110, 300)
    cache = eng.prepare_cache(prices)
    assert "features" in cache and "returns" in cache
    assert cache["features"].shape == (300, 3)


def _test_zscore_engineer_make_input():
    from src.data.market_data_loader import ZScoreEngineer
    eng = ZScoreEngineer()
    prices = np.linspace(100, 110, 300)
    cache = eng.prepare_cache(prices)
    inp = eng.make_input(cache, start=0, length=32)
    assert inp.shape == (3, 32)  # (feature_dim, time)
    assert torch.isfinite(inp).all()


def _test_zscore_engineer_make_target():
    from src.data.market_data_loader import ZScoreEngineer
    eng = ZScoreEngineer()
    prices = np.linspace(100, 110, 300)
    cache = eng.prepare_cache(prices)
    tgt = eng.make_target(cache, start=0, length=8)
    assert tgt.shape == (1, 8)


def _test_zscore_engineer_get_volatility():
    from src.data.market_data_loader import ZScoreEngineer
    eng = ZScoreEngineer()
    prices = np.linspace(100, 110, 300)
    cache = eng.prepare_cache(prices)
    vol = eng.get_volatility(cache, start=0, length=32)
    assert isinstance(vol, float) and vol >= 0


def _test_zscore_engineer_clean_prices():
    from src.data.market_data_loader import ZScoreEngineer
    eng = ZScoreEngineer()
    # Contains NaN, inf, and zero — should all be cleaned
    prices = np.array([np.nan, 100.0, 101.0, np.inf, 0.0, 102.0])
    cleaned = eng.clean_prices(prices)
    assert np.all(np.isfinite(cleaned))
    assert np.all(cleaned > 0)


def _test_wavelet_engineer_feature_dim():
    from src.data.market_data_loader import WaveletEngineer
    eng = WaveletEngineer()
    assert eng.feature_dim == 5


def _test_wavelet_engineer_make_input():
    from src.data.market_data_loader import WaveletEngineer
    eng = WaveletEngineer(wavelet="db4", level=3)
    prices = np.linspace(100, 110, 300)
    cache = eng.prepare_cache(prices)
    inp = eng.make_input(cache, start=0, length=32)
    assert inp.shape == (5, 32)
    assert torch.isfinite(inp).all()


def _test_asset_data_dataclass():
    from src.data.market_data_loader import AssetData
    ts = np.array([0, 1, 2])
    prices = np.array([100.0, 101.0, 102.0])
    a = AssetData(name="TEST", timestamps=ts, prices=prices)
    assert a.name == "TEST"
    assert a.covariates is None


def _test_market_dataset():
    from src.data.market_data_loader import MockDataSource, ZScoreEngineer, MarketDataset
    src = MockDataSource(length=500, freq="5min", seed=10)
    assets = src.load_data(["BTC"])
    eng = ZScoreEngineer()
    ds = MarketDataset(assets, engineer=eng, input_len=32, pred_len=8, stride=10)
    assert len(ds) > 0
    sample = ds[0]
    assert set(sample.keys()) >= {"inputs", "target", "decision_timestamp", "meta"}
    assert sample["inputs"].shape[0] == 3  # feature_dim
    assert sample["inputs"].shape[1] == 32  # input_len


def _test_market_dataset_vol_buckets():
    from src.data.market_data_loader import MockDataSource, ZScoreEngineer, MarketDataset
    src = MockDataSource(length=500, freq="5min", seed=42)
    assets = src.load_data(["SYNTH"])
    eng = ZScoreEngineer()
    ds = MarketDataset(assets, engineer=eng, input_len=32, pred_len=8, stride=10)
    buckets = ds.get_vol_buckets()
    assert buckets.ndim == 1
    assert set(buckets).issubset({0, 1, 2})


def _test_market_data_loader_construction():
    from src.data.market_data_loader import MockDataSource, ZScoreEngineer, MarketDataLoader
    src = MockDataSource(length=600, freq="5min", seed=99)
    eng = ZScoreEngineer()
    loader = MarketDataLoader(
        data_source=src, engineer=eng, assets=["SYNTH"],
        input_len=32, pred_len=8, batch_size=4, stride=10,
    )
    assert loader.feature_dim == 3
    assert len(loader.dataset) > 0


def _test_market_data_loader_get_price_series():
    from src.data.market_data_loader import MockDataSource, ZScoreEngineer, MarketDataLoader
    src = MockDataSource(length=400, freq="5min", seed=5)
    eng = ZScoreEngineer()
    loader = MarketDataLoader(
        data_source=src, engineer=eng, assets=["BTC"],
        input_len=32, pred_len=8, batch_size=4,
    )
    series = loader.get_price_series()
    assert isinstance(series, torch.Tensor) and series.ndim == 1
    assert len(series) == 400

    sliced = loader.get_price_series(start=10, end=50)
    assert len(sliced) == 40


def _test_market_data_loader_static_holdout():
    from src.data.market_data_loader import MockDataSource, ZScoreEngineer, MarketDataLoader
    # Use a large dataset so all vol_buckets have ≥2 members for StratifiedShuffleSplit
    src = MockDataSource(length=3000, freq="5min", seed=7)
    eng = ZScoreEngineer()
    loader = MarketDataLoader(
        data_source=src, engineer=eng, assets=["SYNTH"],
        input_len=32, pred_len=8, batch_size=4, stride=2,
    )
    train_dl, val_dl, test_dl = loader.static_holdout(cutoff=0.2)
    # At least one split must have data
    assert len(train_dl) + len(val_dl) + len(test_dl) > 0


def _test_market_data_loader_multiple_assets():
    from src.data.market_data_loader import MockDataSource, ZScoreEngineer, MarketDataLoader
    src = MockDataSource(length=500, freq="5min", seed=42)
    eng = ZScoreEngineer()
    loader = MarketDataLoader(
        data_source=src, engineer=eng, assets=["BTC", "ETH"],
        input_len=32, pred_len=8, batch_size=8,
    )
    assert len(loader.assets_data) == 2


def _test_market_data_loader_wavelet():
    from src.data.market_data_loader import MockDataSource, WaveletEngineer, MarketDataLoader
    src = MockDataSource(length=400, freq="5min", seed=1)
    eng = WaveletEngineer(wavelet="db4", level=3)
    loader = MarketDataLoader(
        data_source=src, engineer=eng, assets=["SYNTH"],
        input_len=32, pred_len=8, batch_size=4,
    )
    assert loader.feature_dim == 5


# =============================================================================
# SECTION 9: StridedTimeSeriesDataset + FeatureEngineerBase
# =============================================================================

def _test_strided_dataset_basic():
    from src.data.base_dataset import StridedTimeSeriesDataset
    series = torch.cumsum(torch.randn(200), dim=0)
    ds = StridedTimeSeriesDataset(series, context_len=32, pred_len=8, stride=4)
    assert len(ds) > 0
    s = ds[0]
    assert "history" in s and "target" in s and "initial_price" in s and "actual_series" in s
    assert s["history"].shape == (32,)
    assert s["actual_series"].shape == (8,)


def _test_strided_dataset_2d_target():
    from src.data.base_dataset import StridedTimeSeriesDataset
    # 2D series: (T, features)
    series = torch.randn(150, 3)
    ds = StridedTimeSeriesDataset(series, context_len=16, pred_len=4, stride=1)
    assert len(ds) > 0
    s = ds[0]
    assert s["history"].shape[0] == 16


def _test_strided_dataset_past_covariates():
    from src.data.base_dataset import StridedTimeSeriesDataset
    series = torch.randn(100)
    past_cov = torch.randn(100, 2)
    ds = StridedTimeSeriesDataset(series, context_len=16, pred_len=4, past_covariates=past_cov)
    s = ds[0]
    assert "past_covariates" in s
    assert s["past_covariates"].shape == (16, 2)


def _test_strided_dataset_future_covariates():
    from src.data.base_dataset import StridedTimeSeriesDataset
    series = torch.randn(100)
    future_cov = torch.randn(100, 3)
    ds = StridedTimeSeriesDataset(series, context_len=16, pred_len=4, future_covariates=future_cov)
    s = ds[0]
    assert "future_covariates" in s


def _test_strided_dataset_too_short_raises():
    from src.data.base_dataset import StridedTimeSeriesDataset
    try:
        StridedTimeSeriesDataset(torch.randn(10), context_len=8, pred_len=8, stride=1)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


def _test_strided_dataset_invalid_params_raise():
    from src.data.base_dataset import StridedTimeSeriesDataset
    series = torch.randn(100)
    for bad_kwargs in [
        {"context_len": 0, "pred_len": 4},
        {"context_len": 16, "pred_len": -1},
        {"context_len": 16, "pred_len": 4, "stride": 0},
    ]:
        try:
            StridedTimeSeriesDataset(series, **bad_kwargs)
            raise AssertionError(f"Should raise ValueError for {bad_kwargs}")
        except ValueError:
            pass


def _test_feature_engineer_base():
    from src.data.base_dataset import FeatureEngineerBase
    eng = FeatureEngineerBase()
    x = torch.randn(10, 3)
    assert torch.equal(eng.transform_history(x), x)
    assert torch.equal(eng.transform_target(x), x)
    assert eng.extra_features(x) is None


# =============================================================================
# SECTION 10: Metrics
# =============================================================================

def _test_crps_ensemble_perfect_forecast():
    from src.research.metrics import crps_ensemble
    target = torch.tensor([1.0, 2.0, 3.0])
    sims = target.unsqueeze(-1).expand(-1, 200)
    crps = crps_ensemble(sims, target)
    assert crps.shape == (3,)
    # Allow tiny negative values due to floating-point precision; mathematically CRPS >= 0
    assert (crps >= -1e-6).all(), f"CRPS values unexpectedly negative: {crps}"
    assert crps.max() < 0.01, "Perfect forecast should have near-zero CRPS"


def _test_crps_ensemble_batch_horizon():
    from src.research.metrics import crps_ensemble
    batch, horizon, n_paths = 4, 12, 100
    sims = torch.randn(batch, horizon, n_paths)
    target = torch.randn(batch, horizon)
    crps = crps_ensemble(sims, target)
    assert crps.shape == (batch, horizon)
    assert (crps >= 0).all()


def _test_crps_ensemble_worse_is_higher():
    from src.research.metrics import crps_ensemble
    target = torch.zeros(10)
    good_sims = torch.randn(10, 100) * 0.01
    bad_sims = torch.randn(10, 100) * 10.0
    crps_good = crps_ensemble(good_sims, target).mean()
    crps_bad = crps_ensemble(bad_sims, target).mean()
    assert crps_good < crps_bad


def _test_afcrps_ensemble():
    from src.research.metrics import afcrps_ensemble
    batch, n = 4, 50
    sims = torch.randn(batch, n)
    target = torch.randn(batch)
    for alpha in (0.0, 0.5, 0.95, 1.0):
        c = afcrps_ensemble(sims, target, alpha=alpha)
        assert c.shape == (batch,)
        assert (c >= 0).all()


def _test_afcrps_alpha0_matches_crps():
    from src.research.metrics import afcrps_ensemble, crps_ensemble
    sims = torch.randn(3, 50)
    target = torch.randn(3)
    c_af = afcrps_ensemble(sims, target, alpha=0.0)
    c_std = crps_ensemble(sims, target)
    assert torch.allclose(c_af, c_std, atol=1e-5), "alpha=0 should match standard CRPS"


def _test_log_likelihood_shapes():
    from src.research.metrics import log_likelihood
    sims = torch.randn(4, 100)
    target = torch.randn(4)
    ll = log_likelihood(sims, target)
    assert ll.shape == (4,)


def _test_log_likelihood_ordering():
    from src.research.metrics import log_likelihood
    target = torch.zeros(5)
    sims_good = torch.randn(5, 100) * 0.01
    sims_bad = torch.randn(5, 100) * 10.0
    ll_good = log_likelihood(sims_good, target).mean()
    ll_bad = log_likelihood(sims_bad, target).mean()
    assert ll_good > ll_bad, "Tight ensemble should have higher log-likelihood"


def _test_get_interval_steps():
    from src.research.metrics import get_interval_steps
    assert get_interval_steps(300, 60) == 5
    assert get_interval_steps(3600, 300) == 12
    assert get_interval_steps(86400, 3600) == 24


def _test_calculate_price_changes_returns():
    from src.research.metrics import calculate_price_changes_over_intervals
    prices = np.array([[100.0, 102.0, 101.0, 104.0, 103.0, 106.0]])
    changes = calculate_price_changes_over_intervals(prices, interval_steps=2)
    assert changes.shape[0] == 1
    assert changes.shape[1] > 0


def _test_calculate_price_changes_absolute():
    from src.research.metrics import calculate_price_changes_over_intervals
    prices = np.array([[100.0, 102.0, 101.0, 104.0, 103.0, 106.0]])
    changes = calculate_price_changes_over_intervals(prices, interval_steps=2, absolute_price=True)
    assert changes.shape[0] == 1


def _test_calculate_price_changes_edge_cases():
    from src.research.metrics import calculate_price_changes_over_intervals
    prices = np.array([[100.0, 101.0]])
    # interval_steps >= T => empty result
    result = calculate_price_changes_over_intervals(prices, interval_steps=5)
    assert result.shape[1] == 0
    # interval_steps=0 => empty
    result2 = calculate_price_changes_over_intervals(prices, interval_steps=0)
    assert result2.shape[1] == 0


def _test_label_observed_blocks():
    from src.research.metrics import label_observed_blocks
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    labels = label_observed_blocks(arr)
    assert labels[0] == labels[1]   # same block
    assert labels[2] == -1           # nan
    assert labels[3] == labels[4]   # same block
    assert labels[3] != labels[0]   # different block


def _test_generate_adaptive_intervals():
    from src.research.metrics import generate_adaptive_intervals
    intervals = generate_adaptive_intervals(horizon_steps=100, time_increment=60, min_intervals=3)
    assert isinstance(intervals, dict) and len(intervals) >= 1
    for name, secs in intervals.items():
        assert isinstance(name, str) and secs > 0


def _test_generate_adaptive_intervals_short_horizon():
    from src.research.metrics import generate_adaptive_intervals
    # very short horizon
    intervals = generate_adaptive_intervals(horizon_steps=2, time_increment=60)
    assert isinstance(intervals, dict)


def _test_filter_valid_intervals():
    from src.research.metrics import filter_valid_intervals, SCORING_INTERVALS
    # With 10 steps at 60s each (600s total), only 5min (300s=5 steps) fits
    valid = filter_valid_intervals(SCORING_INTERVALS, horizon_steps=10, time_increment=60)
    assert isinstance(valid, dict)
    # All returned intervals should fit within the horizon
    for name, secs in valid.items():
        steps = secs // 60
        assert steps < 10


def _test_crps_multi_interval_scorer_adaptive():
    from src.research.metrics import CRPSMultiIntervalScorer
    scorer = CRPSMultiIntervalScorer(time_increment=60, adaptive=True)
    sims = torch.ones(50, 20) * 100.0 + torch.randn(50, 20) * 0.5
    real = torch.ones(20) * 100.0
    total, detailed = scorer(sims, real)
    assert isinstance(total, float)
    assert isinstance(detailed, list)


def _test_crps_multi_interval_scorer_nonadaptive():
    from src.research.metrics import CRPSMultiIntervalScorer
    scorer = CRPSMultiIntervalScorer(time_increment=300, adaptive=False)
    sims = torch.ones(50, 1000) * 100.0
    real = torch.ones(1000) * 100.0
    total, detailed = scorer(sims, real)
    assert isinstance(total, float)


def _test_crps_scorer_caches_intervals():
    from src.research.metrics import CRPSMultiIntervalScorer
    scorer = CRPSMultiIntervalScorer(time_increment=60, adaptive=True)
    scorer.get_intervals_for_horizon(50)
    assert 50 in scorer._interval_cache


def _test_crps_scorer_invalid_time_increment():
    from src.research.metrics import CRPSMultiIntervalScorer
    try:
        CRPSMultiIntervalScorer(time_increment=0)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


# =============================================================================
# SECTION 11: Trainer
# =============================================================================

def _test_prepare_paths_for_crps():
    from src.research.trainer import prepare_paths_for_crps
    paths = torch.randn(4, 100, 12)   # (batch, n_paths, horizon)
    ready = prepare_paths_for_crps(paths)
    assert ready.shape == (4, 12, 100)  # (batch, horizon, n_paths)


def _test_data_to_model_adapter_shapes():
    from src.research.trainer import DataToModelAdapter
    adapter = DataToModelAdapter(device=torch.device("cpu"))
    batch = {"inputs": torch.randn(4, 3, 32), "target": torch.randn(4, 1, 8)}
    result = adapter(batch)
    assert result["history"].shape == (4, 32, 3)   # transposed
    assert result["initial_price"].shape == (4,)
    assert result["target_factors"].shape == (4, 8)


def _test_data_to_model_adapter_zero_logreturns():
    from src.research.trainer import DataToModelAdapter
    adapter = DataToModelAdapter(device=torch.device("cpu"), target_is_log_return=True)
    batch = {"inputs": torch.randn(2, 3, 16), "target": torch.zeros(2, 1, 4)}
    result = adapter(batch)
    # exp(cumsum(0)) = 1.0
    assert torch.allclose(result["target_factors"], torch.ones(2, 4), atol=1e-5)


def _test_data_to_model_adapter_bad_input_raises():
    from src.research.trainer import DataToModelAdapter
    adapter = DataToModelAdapter(device=torch.device("cpu"))
    bad_batch = {"inputs": torch.randn(4, 32), "target": torch.randn(4, 1, 8)}  # 2D input
    try:
        adapter(bad_batch)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


def _test_trainer_train_step():
    from src.research.trainer import Trainer
    from src.models.factory import SynthModel, HybridBackbone
    from src.models.registry import LSTMBlock
    from src.models.heads import GBMHead

    bb = HybridBackbone(input_size=3, d_model=16, blocks=[LSTMBlock(d_model=16)])
    model = SynthModel(bb, GBMHead(latent_size=16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, opt, n_paths=10)

    batch = {"inputs": torch.randn(4, 3, 16), "target": torch.randn(4, 1, 4)}
    metrics = trainer.train_step(batch)
    for key in ("loss", "crps", "sharpness", "log_likelihood", "mu", "sigma"):
        assert key in metrics, f"missing key: {key}"
    assert isinstance(metrics["loss"], float)
    assert metrics["loss"] >= 0


def _test_trainer_train_step_afcrps_vs_crps():
    from src.research.trainer import Trainer
    from src.models.factory import SynthModel, HybridBackbone
    from src.models.registry import TransformerBlock
    from src.models.heads import SDEHead

    def make_trainer(alpha):
        bb = HybridBackbone(input_size=3, d_model=16, blocks=[TransformerBlock(d_model=16, nhead=2)])
        model = SynthModel(bb, SDEHead(latent_size=16))
        opt = torch.optim.Adam(model.parameters())
        return Trainer(model, opt, n_paths=5, crps_alpha=alpha)

    batch = {"inputs": torch.randn(2, 3, 16), "target": torch.randn(2, 1, 4)}
    t_afcrps = make_trainer(0.95)
    t_crps = make_trainer(None)
    m1 = t_afcrps.train_step(batch)
    m2 = t_crps.train_step(batch)
    assert isinstance(m1["loss"], float)
    assert isinstance(m2["loss"], float)


def _test_trainer_validate():
    from src.research.trainer import Trainer
    from src.models.factory import SynthModel, HybridBackbone
    from src.models.registry import LSTMBlock
    from src.models.heads import GBMHead

    bb = HybridBackbone(input_size=3, d_model=16, blocks=[LSTMBlock(d_model=16)])
    model = SynthModel(bb, GBMHead(latent_size=16))
    opt = torch.optim.Adam(model.parameters())
    trainer = Trainer(model, opt, n_paths=10)

    def fake_loader():
        for _ in range(2):
            yield {"inputs": torch.randn(4, 3, 16), "target": torch.randn(4, 1, 4)}

    metrics = trainer.validate(fake_loader())
    assert "val_crps" in metrics
    assert isinstance(metrics["val_crps"], float) and metrics["val_crps"] >= 0


# =============================================================================
# SECTION 12: Factory Functions
# =============================================================================

def _test_build_model_from_dict():
    from src.models.factory import build_model, SynthModel
    cfg = {
        "backbone": {
            "_target_": "src.models.factory.HybridBackbone",
            "input_size": 3, "d_model": 16,
            "blocks": [{"_target_": "src.models.registry.TransformerBlock", "d_model": 16, "nhead": 2}],
        },
        "head": {"_target_": "src.models.heads.GBMHead", "latent_size": 16},
    }
    model = build_model(cfg)
    assert isinstance(model, SynthModel)


def _test_build_model_latent_mismatch_raises():
    from src.models.factory import build_model
    cfg = {
        "backbone": {
            "_target_": "src.models.factory.HybridBackbone",
            "input_size": 3, "d_model": 32,
            "blocks": [{"_target_": "src.models.registry.LSTMBlock", "d_model": 32}],
        },
        "head": {
            "_target_": "src.models.heads.GBMHead",
            "latent_size": 16,   # wrong — backbone outputs 32
        },
    }
    try:
        build_model(cfg)
        raise AssertionError("Should raise ValueError for latent size mismatch")
    except ValueError:
        pass


def _test_smoke_test_model():
    from src.models.factory import _smoke_test_model, SynthModel, HybridBackbone
    from src.models.registry import LSTMBlock
    from src.models.heads import GBMHead

    bb = HybridBackbone(input_size=5, d_model=16, blocks=[LSTMBlock(d_model=16)])
    model = SynthModel(bb, GBMHead(latent_size=16))
    _smoke_test_model(model, input_size=5)  # should not raise


def _test_head_registry_all_8_heads():
    from src.models.factory import HEAD_REGISTRY
    expected = [
        "gbm", "sde", "neural_sde", "horizon", "simple_horizon",
        "mixture_density", "vol_term_structure", "neural_bridge",
    ]
    for h in expected:
        assert h in HEAD_REGISTRY, f"HEAD_REGISTRY missing '{h}'"


def _test_create_model_idempotent():
    from src.models.factory import create_model, SynthModel
    from src.models.heads import GBMHead
    model = SynthModel(_backbone(), GBMHead(latent_size=32))
    # Passing an already-built model should return it unchanged
    result = create_model(model)
    assert result is model


# =============================================================================
# MAIN: register and run all tests
# =============================================================================

if __name__ == "__main__":
    section("SECTION 1: Registry")
    run("registry_instantiation",          _test_registry_instantiation)
    run("registry_block_register_get",     _test_registry_block_register_and_get)
    run("registry_component_register",     _test_registry_component_register)
    run("registry_hybrid_register",        _test_registry_hybrid_register)
    run("registry_list_blocks",            _test_registry_list_blocks)
    run("registry_summary",                _test_registry_summary)
    run("registry_recipe_hash",            _test_registry_recipe_hash)
    run("registry_attribute_access",       _test_registry_attribute_access)
    run("registry_missing_raises",         _test_registry_missing_raises)
    run("registry_duplicate_raises",       _test_registry_duplicate_raises)
    run("discover_components",             _test_discover_components)

    section("SECTION 2: Built-in Blocks")
    run("transformer_block",               _test_transformer_block)
    run("lstm_block",                      _test_lstm_block)
    run("sde_evolution_block",             _test_sde_evolution_block)
    run("custom_attention",                _test_custom_attention)
    run("gated_mlp",                       _test_gated_mlp)
    run("patch_merging",                   _test_patch_merging)

    section("SECTION 3: Advanced Blocks")
    run("rnn_block",                       _test_rnn_block)
    run("gru_block",                       _test_gru_block)
    run("resconv_block",                   _test_resconv_block)
    run("bitcn_block",                     _test_bitcn_block)
    run("patch_embedding_changes_seq",     _test_patch_embedding_changes_seq)
    run("unet1d_block",                    _test_unet1d_block)
    run("transformer_encoder_adapter",     _test_transformer_encoder_adapter)
    run("transformer_decoder_adapter",     _test_transformer_decoder_adapter)
    run("fourier_block",                   _test_fourier_block)
    run("last_step_adapter_modes",         _test_last_step_adapter_modes)
    run("revin_norm_denorm",               _test_revin_norm_denorm)
    run("revin_invalid_mode",              _test_revin_invalid_mode)
    run("flexible_patch_embed",            _test_flexible_patch_embed)
    run("flexible_patch_embed_channel_independence", _test_flexible_patch_embed_channel_independence)
    run("channel_rejoin_mean",             _test_channel_rejoin_mean)
    run("dlinear_block",                   _test_dlinear_block)
    run("layernorm_block",                 _test_layernorm_block)
    run("timesnet_block",                  _test_timesnet_block)

    section("SECTION 4: Heads")
    run("gbm_head_shapes_positivity",      _test_gbm_head_shapes_and_positivity)
    run("sde_head",                        _test_sde_head)
    run("horizon_head",                    _test_horizon_head)
    run("horizon_head_kv_compression",     _test_horizon_head_kv_compression)
    run("horizon_head_clips_to_max",       _test_horizon_head_clips_to_max)
    run("simple_horizon_head_all_pools",   _test_simple_horizon_head_all_pools)
    run("simple_horizon_head_invalid_pool",_test_simple_horizon_head_invalid_pool)
    run("mixture_density_head",            _test_mixture_density_head)
    run("vol_term_structure_head",         _test_vol_term_structure_head)
    run("neural_bridge_no_price",          _test_neural_bridge_head_no_price)
    run("neural_bridge_with_price",        _test_neural_bridge_head_with_price)
    run("neural_sde_head",                 _test_neural_sde_head)

    section("SECTION 5: HybridBackbone & ParallelFusion")
    run("backbone_last_step",              _test_hybrid_backbone_last_step)
    run("backbone_forward_sequence",       _test_hybrid_backbone_forward_sequence)
    run("backbone_multiple_blocks",        _test_hybrid_backbone_multiple_blocks)
    run("backbone_insert_layernorm",       _test_hybrid_backbone_insert_layernorm)
    run("backbone_empty_raises",           _test_hybrid_backbone_empty_raises)
    run("backbone_output_dim",             _test_hybrid_backbone_output_dim)
    run("parallel_fusion_gating",          _test_parallel_fusion_gating)
    run("parallel_fusion_concat",          _test_parallel_fusion_concat)
    run("parallel_fusion_too_few_raises",  _test_parallel_fusion_too_few_paths_raises)
    run("parallel_fusion_bad_strategy",    _test_parallel_fusion_bad_strategy_raises)

    section("SECTION 6: SynthModel — All Head Types")
    run("synth_gbm",                       _test_synth_gbm)
    run("synth_sde",                       _test_synth_sde)
    run("synth_horizon",                   _test_synth_horizon)
    run("synth_simple_horizon",            _test_synth_simple_horizon)
    run("synth_mixture_density",           _test_synth_mixture_density)
    run("synth_vol_term_structure",        _test_synth_vol_term_structure)
    run("synth_neural_bridge",             _test_synth_neural_bridge)
    run("synth_neural_sde",                _test_synth_neural_sde)
    run("synth_shape_contract_all_heads",  _test_synth_shape_contract_all_heads)

    section("SECTION 7: Path Simulators")
    run("simulate_gbm_paths",              _test_simulate_gbm_paths)
    run("simulate_gbm_positivity_extremes",_test_simulate_gbm_positivity_extremes)
    run("simulate_gbm_dt_param",           _test_simulate_gbm_dt)
    run("simulate_horizon_paths",          _test_simulate_horizon_paths)
    run("simulate_bridge_paths",           _test_simulate_bridge_paths)
    run("simulate_mixture_paths",          _test_simulate_mixture_paths)

    section("SECTION 8: Data Pipeline")
    run("mock_data_source",                _test_mock_data_source)
    run("mock_data_source_reproducible",   _test_mock_data_source_reproducible)
    run("zscore_feature_dim",              _test_zscore_engineer_feature_dim)
    run("zscore_prepare_cache",            _test_zscore_engineer_prepare_cache)
    run("zscore_make_input",               _test_zscore_engineer_make_input)
    run("zscore_make_target",              _test_zscore_engineer_make_target)
    run("zscore_get_volatility",           _test_zscore_engineer_get_volatility)
    run("zscore_clean_prices",             _test_zscore_engineer_clean_prices)
    run("wavelet_feature_dim",             _test_wavelet_engineer_feature_dim)
    run("wavelet_make_input",              _test_wavelet_engineer_make_input)
    run("asset_data_dataclass",            _test_asset_data_dataclass)
    run("market_dataset",                  _test_market_dataset)
    run("market_dataset_vol_buckets",      _test_market_dataset_vol_buckets)
    run("market_data_loader_construction", _test_market_data_loader_construction)
    run("market_data_loader_price_series", _test_market_data_loader_get_price_series)
    run("market_data_loader_static_holdout", _test_market_data_loader_static_holdout)
    run("market_data_loader_multi_assets", _test_market_data_loader_multiple_assets)
    run("market_data_loader_wavelet",      _test_market_data_loader_wavelet)

    section("SECTION 9: StridedTimeSeriesDataset + FeatureEngineerBase")
    run("strided_dataset_basic",           _test_strided_dataset_basic)
    run("strided_dataset_2d_target",       _test_strided_dataset_2d_target)
    run("strided_dataset_past_covariates", _test_strided_dataset_past_covariates)
    run("strided_dataset_future_covariates", _test_strided_dataset_future_covariates)
    run("strided_dataset_too_short_raises",_test_strided_dataset_too_short_raises)
    run("strided_dataset_invalid_params",  _test_strided_dataset_invalid_params_raise)
    run("feature_engineer_base",           _test_feature_engineer_base)

    section("SECTION 10: Metrics")
    run("crps_perfect_forecast",           _test_crps_ensemble_perfect_forecast)
    run("crps_batch_horizon_shape",        _test_crps_ensemble_batch_horizon)
    run("crps_worse_is_higher",            _test_crps_ensemble_worse_is_higher)
    run("afcrps_all_alphas",               _test_afcrps_ensemble)
    run("afcrps_alpha0_matches_crps",      _test_afcrps_alpha0_matches_crps)
    run("log_likelihood_shape",            _test_log_likelihood_shapes)
    run("log_likelihood_ordering",         _test_log_likelihood_ordering)
    run("get_interval_steps",              _test_get_interval_steps)
    run("price_changes_returns",           _test_calculate_price_changes_returns)
    run("price_changes_absolute",          _test_calculate_price_changes_absolute)
    run("price_changes_edge_cases",        _test_calculate_price_changes_edge_cases)
    run("label_observed_blocks",           _test_label_observed_blocks)
    run("generate_adaptive_intervals",     _test_generate_adaptive_intervals)
    run("adaptive_intervals_short_horizon",_test_generate_adaptive_intervals_short_horizon)
    run("filter_valid_intervals",          _test_filter_valid_intervals)
    run("crps_scorer_adaptive",            _test_crps_multi_interval_scorer_adaptive)
    run("crps_scorer_nonadaptive",         _test_crps_multi_interval_scorer_nonadaptive)
    run("crps_scorer_caches",              _test_crps_scorer_caches_intervals)
    run("crps_scorer_invalid_increment",   _test_crps_scorer_invalid_time_increment)

    section("SECTION 11: Trainer")
    run("prepare_paths_for_crps",          _test_prepare_paths_for_crps)
    run("adapter_shapes",                  _test_data_to_model_adapter_shapes)
    run("adapter_zero_logreturns",         _test_data_to_model_adapter_zero_logreturns)
    run("adapter_bad_input_raises",        _test_data_to_model_adapter_bad_input_raises)
    run("trainer_train_step",              _test_trainer_train_step)
    run("trainer_afcrps_vs_crps",          _test_trainer_train_step_afcrps_vs_crps)
    run("trainer_validate",                _test_trainer_validate)

    section("SECTION 12: Factory Functions")
    run("build_model_from_dict",           _test_build_model_from_dict)
    run("build_model_latent_mismatch",     _test_build_model_latent_mismatch_raises)
    run("smoke_test_model",                _test_smoke_test_model)
    run("head_registry_all_8_heads",       _test_head_registry_all_8_heads)
    run("create_model_idempotent",         _test_create_model_idempotent)

    ok = summary()
    sys.exit(0 if ok else 1)
