"""Structured logging facade."""

from __future__ import annotations

from flext_core import FlextTypes as t

from ._loggings_parts import FlextLogger

__all__: t.StrSequence = ("FlextLogger",)
