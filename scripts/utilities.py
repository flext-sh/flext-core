"""Utility functions for flext."""

from __future__ import annotations

from flext_core import u
from scripts import t


class ScriptsFlextUtilities(u):
    """Utility functions for flext."""


u = ScriptsFlextUtilities

__all__: t.MutableSequenceOf[str] = ["ScriptsFlextUtilities", "u"]
