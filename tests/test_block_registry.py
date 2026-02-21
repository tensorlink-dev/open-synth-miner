"""Tests for the block registry introspection features."""
import dataclasses

import pytest

from src.models.registry import BlockInfo, Registry, discover_components, registry


def test_core_blocks_have_info():
    """Core blocks defined in registry.py should have BlockInfo entries."""
    for name in ("transformerblock", "lstmblock", "sdeevolutionblock"):
        info = registry.get_info(name)
        assert isinstance(info, BlockInfo)
        assert info.kind == "block"
        assert info.description  # non-empty


def test_core_components_have_info():
    """Components defined in registry.py should have BlockInfo entries."""
    for name in ("customattention", "gatedmlp", "patchmerging"):
        info = registry.get_info(name)
        assert isinstance(info, BlockInfo)
        assert info.kind == "component"
        assert info.description


def test_hybrid_has_info():
    info = registry.get_info("attn_sde")
    assert info.kind == "hybrid"
    assert info.description


def test_advanced_blocks_registered_after_discovery():
    """After discover_components (run by conftest session fixture), advanced blocks appear."""
    # The session fixture in conftest.py ensures discovery already ran.
    advanced_names = [
        "rnnblock", "grublock", "resconvblock", "bitcnblock",
        "patchembedding", "unet1dblock", "transformerencoder",
        "transformerdecoder", "fourierblock", "laststepadapter",
        "revin", "flexiblepatchembed", "channelrejoin",
        "multiscalepatcher", "dlinearblock", "layernormblock",
        "timesnetblock", "timemixerblock", "patchmixerblock",
    ]
    for name in advanced_names:
        info = registry.get_info(name)
        assert info.kind == "block", f"{name} should be a block"
        assert info.description, f"{name} should have a description"


def test_discover_components_is_idempotent():
    """discover_components can be called multiple times without raising or duplicating entries."""
    before = len(registry.blocks)
    discover_components("src/models/components")
    after = len(registry.blocks)
    # Because Python caches module imports the decorators don't re-run,
    # so the count must stay the same (no duplicates, no KeyError).
    assert after == before, (
        f"Second discover_components call changed block count: {before} -> {after}"
    )


def test_list_blocks_returns_all():
    all_entries = registry.list_blocks()
    assert len(all_entries) >= 22  # 3 components + 19 advanced + 3 core blocks + 1 hybrid


def test_list_blocks_filter_by_kind():
    blocks_only = registry.list_blocks(kind="block")
    assert all(e.kind == "block" for e in blocks_only)
    assert len(blocks_only) >= 19

    components_only = registry.list_blocks(kind="component")
    assert all(e.kind == "component" for e in components_only)
    assert len(components_only) >= 3

    hybrids_only = registry.list_blocks(kind="hybrid")
    assert all(e.kind == "hybrid" for e in hybrids_only)
    assert len(hybrids_only) >= 1


def test_get_info_preserves_shape_metadata():
    info = registry.get_info("transformerblock")
    assert info.preserves_seq_len is True
    assert info.preserves_d_model is True
    assert info.min_seq_len == 1

    patch_info = registry.get_info("patchembedding")
    assert patch_info.preserves_seq_len is False

    timesnet_info = registry.get_info("timesnetblock")
    assert timesnet_info.min_seq_len == 4


def test_get_info_unknown_raises():
    with pytest.raises(KeyError):
        registry.get_info("nonexistent_block")


def test_summary_produces_table():
    table = registry.summary()
    assert "Name" in table
    assert "Kind" in table
    assert "Description" in table
    assert "transformerblock" in table
    assert "lstmblock" in table


def test_summary_filter_by_kind():
    table = registry.summary(kind="component")
    assert "component" in table
    assert "block" not in table.split("\n", 2)[-1]  # not in data rows


def test_summary_empty_registry():
    empty = Registry()
    assert empty.summary() == "No registered entries."


def test_blockinfo_is_frozen():
    info = registry.get_info("transformerblock")
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.name = "hacked"
