"""Rule 1 violation: loose Final constant outside Constants class."""

from __future__ import annotations

from typing import Final


class FlextTestModels:
    """Models namespace."""

    class Inner:
        """Inner class."""

        pass


MAX_RETRIES: Final[int] = 3  # VIOLATION — should be in Constants
DEFAULT_TIMEOUT: Final[float] = 30.0  # VIOLATION
