"""Constants for flext."""

from __future__ import annotations

from flext_core import FlextConstants
from scripts import t


class ScriptsFlextConstants(FlextConstants):
    """Constants for flext."""


c = ScriptsFlextConstants

__all__: t.MutableSequenceOf[str] = ["ScriptsFlextConstants", "c"]
