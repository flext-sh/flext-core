"""Domain models for flext."""

from __future__ import annotations

from flext_cli import m
from scripts import t


class ScriptsFlextModels(m):
    """Domain models for flext."""


m = ScriptsFlextModels

__all__: t.MutableSequenceOf[str] = ["ScriptsFlextModels", "m"]
