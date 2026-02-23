"""Alias package that re-exports the public Open Synth Miner surface.

This allows ``import open_synth_miner`` when the project is installed
or when the repository root is on the Python path.
"""

import osa as _osa

from osa import *  # noqa: F401,F403

__all__ = _osa.__all__  # type: ignore[attr-defined]
