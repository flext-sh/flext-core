"""Structured logging facade."""

from __future__ import annotations

from ._loggings_parts.flextlogger_part_05 import FlextUtilitiesLogging
from .typings import FlextTypes as t

__all__: t.StrSequence = ("FlextUtilitiesLogging",)
