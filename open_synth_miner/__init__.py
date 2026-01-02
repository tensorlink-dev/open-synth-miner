"""Alias package that re-exports the public Open Synth Miner surface.

This allows ``import open_synth_miner`` when the project is installed
or when the repository root is on the Python path.
"""

import src as _src

from src import *  # noqa: F401,F403

__all__ = _src.__all__  # type: ignore[attr-defined]
