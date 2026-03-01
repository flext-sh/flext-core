"""Rule 1 violation: method defined inside Constants class."""

from __future__ import annotations

from typing import Final


class FlextTestConstants:
    """Constants with forbidden methods."""

    class Factory:
        """Factory inner class."""

        DEFAULT_NAME: Final[str] = "test"

        @classmethod
        def create_name(cls, prefix: str) -> str:
            """VIOLATION — no methods in Constants."""
            return f"{prefix}_{cls.DEFAULT_NAME}"


c = FlextTestConstants
__all__ = ["FlextTestConstants", "c"]
