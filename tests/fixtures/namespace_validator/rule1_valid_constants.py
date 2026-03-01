"""Rule 1 valid: proper Constants class with inner classes and Final annotations."""

from __future__ import annotations

from typing import Final


class FlextTestConstants:
    """Centralized test constants."""

    class Paths:
        """Path constants."""

        SRC_DIR: Final[str] = "src"
        TEST_DIR: Final[str] = "tests"

    class Limits:
        """Limit constants."""

        MAX_RETRIES: Final[int] = 3
        TIMEOUT: Final[float] = 30.0


c = FlextTestConstants
__all__ = ["FlextTestConstants", "c"]
