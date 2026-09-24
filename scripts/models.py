"""Domain models for flext."""

from __future__ import annotations

from flext_core import FlextModels
from scripts import t


class ScriptsFlextModels(FlextModels):
    """Domain models for flext."""


m = ScriptsFlextModels

__all__: t.MutableSequenceOf[str] = ["ScriptsFlextModels", "m"]
