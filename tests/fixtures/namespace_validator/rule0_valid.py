"""Rule 0 valid: single outer class with correct prefix, allowed loose items."""

from __future__ import annotations

from typing import Final

__all__ = ["FlextTestConstants"]


class FlextTestConstants:
    """Test constants."""

    VALUE: Final[int] = 42


c = FlextTestConstants  # alias — allowed
