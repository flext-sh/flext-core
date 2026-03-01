"""Rule 2 valid: TypeVars at module level, proper Types class."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")
E = TypeVar("E")


class FlextTestTypes:
    """Type namespace for tests."""

    pass


t = FlextTestTypes
__all__ = ["FlextTestTypes", "t"]
