"""Utility functions for flext."""

from __future__ import annotations

from flext_core import FlextUtilities
from scripts import t


class ScriptsFlextUtilities(FlextUtilities):
    """Utility functions for flext."""


u = ScriptsFlextUtilities

__all__: t.MutableSequenceOf[str] = ["ScriptsFlextUtilities", "u"]
