"""Test constants for flext-core tests.

Centralized constants for test fixtures, factories, and test data.
Does NOT duplicate src/flext_core/constants.py - only test-specific constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final

from flext_core.constants import FlextConstants


class TestConstants:
    """Centralized test constants following flext-core nested class pattern.

    Reuses production types from FlextConstants to ensure consistency.
    Uses PEP 695 type aliases (Python 3.13+) for type-safe test constants.
    """

    class Paths:
        """Test path constants."""

        TEST_TEMP_PREFIX: Final[str] = "flext_core_test_"

    class Literals:
        """Literal type aliases for test constants (Python 3.13+ PEP 695 pattern).

        These type aliases reuse production Literals from FlextConstants
        to ensure consistency between tests and production code.

        Note: In Python 3.13+, type aliases defined with `type` keyword can be
        referenced directly. For test-specific types, we use PEP 695 syntax.
        """

        # Reuse production Literals by direct reference (Python 3.13+ best practices)
        # These are type aliases from production, accessible via attribute access
        # Log level literal (reusing production type)
        type LogLevelLiteral = FlextConstants.Literals.LogLevelLiteral

        # Environment literal (reusing production type)
        type EnvironmentLiteral = FlextConstants.Literals.EnvironmentLiteral

        # Registration status literal (reusing production type)
        type RegistrationStatusLiteral = (
            FlextConstants.Literals.RegistrationStatusLiteral
        )

        # Handler mode literal (reusing production type)
        type HandlerModeLiteral = FlextConstants.Literals.HandlerModeLiteral

        # State literal (reusing production type)
        type StateLiteral = FlextConstants.Literals.StateLiteral

        # Context operation literals (reusing production types)
        # Note: These match FlextConstants.Logging type aliases
        type ContextOperationGetLiteral = (
            FlextConstants.Logging.ContextOperationGetLiteral
        )
        type ContextOperationModifyLiteral = (
            FlextConstants.Logging.ContextOperationModifyLiteral
        )

        # CQRS message type literals (reusing production types)
        type CommandMessageTypeLiteral = FlextConstants.Cqrs.CommandMessageTypeLiteral
        type QueryMessageTypeLiteral = FlextConstants.Cqrs.QueryMessageTypeLiteral
        type EventMessageTypeLiteral = FlextConstants.Cqrs.EventMessageTypeLiteral
        type HandlerTypeLiteral = FlextConstants.Cqrs.HandlerTypeLiteral

        # Service metric type literal (reusing production type)
        type ServiceMetricTypeLiteral = FlextConstants.Cqrs.ServiceMetricTypeLiteral

        # Test-specific Literals (not in production)
        # None at this time - all test types reuse production types

    # Note: Type aliases should be imported from flext_core.typings, not constants.
    # Constants file only contains constant values, not type definitions.
    # For types, import directly: from flext_core.typings import FlextTypes


__all__ = ["TestConstants"]
