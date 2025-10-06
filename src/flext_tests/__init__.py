"""FLEXT Testing Framework.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

Testing utilities and fixtures for FLEXT ecosystem projects.
"""

from __future__ import annotations

from flext_core import FlextResult, FlextTypes as FlextTypes


class FlextTestsDomains:
    """FLEXT testing domains for test organization."""

    LDAP = "ldap"
    LDIF = "ldif"
    API = "api"
    CLI = "cli"
    DATABASE = "database"
    MIGRATION = "migration"


class FlextTestsMatchers:
    """FLEXT testing matchers for assertions."""

    @staticmethod
    def is_success_result(result: FlextResult) -> bool:
        """Check if result is successful."""
        return result.is_success

    @staticmethod
    def is_failure_result(result: FlextResult) -> bool:
        """Check if result is a failure."""
        return result.is_failure


# Placeholder for flext_tests module
# This module will be properly implemented in a future version

__version__ = "0.9.9"
