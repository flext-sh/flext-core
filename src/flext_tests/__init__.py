"""FLEXT Testing Framework.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

Testing utilities and fixtures for FLEXT ecosystem projects.
"""

from __future__ import annotations

from flext_core import FlextResult as FlextResult, FlextTypes as FlextTypes


class FlextTestsDomains:
    """FLEXT testing domains for test organization."""

    LDAP = "ldap"
    LDIF = "ldif"
    API = "api"
    CLI = "cli"
    DATABASE = "database"
    MIGRATION = "migration"


# flext-core/src/flext_tests/__init__.py

# (Removed FlextTestsMatchers – callers should use result.is_success / result.is_failure directly.)


# Placeholder for flext_tests module
# This module will be properly implemented in a future version

__version__ = "0.9.9"
