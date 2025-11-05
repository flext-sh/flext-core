"""Utilities module - FlextUtilitiesTypeGuards.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging

# Module constants
MAX_PORT_NUMBER: int = 65535
MIN_PORT_NUMBER: int = 1
_logger = logging.getLogger(__name__)


class FlextUtilitiesTypeGuards:
    """Type guard utilities for runtime type checking."""

    @staticmethod
    def is_string_non_empty(value: object) -> bool:
        """Check if value is a non-empty string."""
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def is_dict_non_empty(value: object) -> bool:
        """Check if value is a non-empty dictionary."""
        return isinstance(value, dict) and bool(value)

    @staticmethod
    def is_list_non_empty(value: object) -> bool:
        """Check if value is a non-empty list."""
        return isinstance(value, list) and bool(value)


__all__ = ["FlextUtilitiesTypeGuards"]
