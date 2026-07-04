"""Structured logging facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._loggings_parts.flextlogger_part_05 import FlextLogger

if TYPE_CHECKING:
    from flext_core import t

__all__: t.StrSequence = ("FlextLogger",)
