"""Shared pytest fixtures for the open-synth-miner test suite."""
import pytest


@pytest.fixture(scope="session", autouse=True)
def _discover_all_components():
    """Import all advanced blocks once per session so the global registry is populated.

    This runs before any test and ensures that tests depending on advanced
    blocks (e.g., rnnblock, grublock) find them in the registry without each
    test needing to call discover_components() individually.
    """
    from src.models.registry import discover_components
    discover_components("src/models/components")
