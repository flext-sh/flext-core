"""Facade for FlextUtilitiesEnforcement."""

from __future__ import annotations

from ._enforcement_parts import FlextUtilitiesEnforcement
from ._enforcement_parts.enforcement_part_01 import PREDICATE_BINDINGS

__all__: list[str] = ["PREDICATE_BINDINGS", "FlextUtilitiesEnforcement"]
