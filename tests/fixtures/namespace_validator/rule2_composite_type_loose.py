"""Rule 2 violation: TypeAlias defined outside typings.py."""

from __future__ import annotations

from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None  # VIOLATION


class FlextTestModels:
    """Models namespace."""

    pass


m = FlextTestModels
__all__ = ["FlextTestModels", "m"]
