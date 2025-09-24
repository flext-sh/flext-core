"""FLEXT Standardization Module.

This module provides standardization utilities for FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.result import FlextResult


class FlextStandardization:
    """FLEXT standardization utilities."""

    @staticmethod
    def standardize_data(data: object) -> FlextResult[object]:
        """Standardize data according to FLEXT patterns.

        Args:
            data: Data to standardize

        Returns:
            FlextResult containing standardized data

        """
        if not data:
            return FlextResult[object].fail("Data cannot be empty")

        return FlextResult[object].ok(data)
