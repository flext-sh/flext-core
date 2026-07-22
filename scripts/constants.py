"""Constants for flext."""

from __future__ import annotations

from flext_cli import c
from scripts import t


class ScriptsFlextConstants(c):
    """Constants for flext."""


c = ScriptsFlextConstants

__all__: t.MutableSequenceOf[str] = ["ScriptsFlextConstants", "c"]
