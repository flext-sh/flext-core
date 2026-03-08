"""Rule 2 violation: TypeAlias defined outside typings.py."""

from __future__ import annotations

from typing import TypeAlias

from flext_core import t

JsonValue: TypeAlias = t.Primitives | None


class FlextTestModels:
    """Models namespace."""

    pass


m = FlextTestModels
__all__ = ["FlextTestModels", "m"]
