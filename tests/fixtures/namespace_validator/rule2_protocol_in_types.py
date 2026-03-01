"""Rule 2 violation: Protocol defined inside Types class."""

from __future__ import annotations

from typing import Protocol


class FlextTestTypes:
    """Types with Protocol inside — WRONG."""

    class Serializable(Protocol):
        """Protocol inside Types — VIOLATION."""

        def to_dict(self) -> dict[str, str]:
            """Convert to dict."""
            ...


t = FlextTestTypes
__all__ = ["FlextTestTypes", "t"]
