"""Constants for FLEXT infra tests.

Provides FlextInfraTestConstants, extending FlextTestsConstants with infra-specific
constants for infrastructure testing, project names, and test markers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final

from tests import TestsFlextConstants


class FlextInfraTestConstants(TestsFlextConstants):
    """Constants for FLEXT infra tests - extends FlextTestsConstants.

    Architecture layer: Layer 0 foundation constants with infra test extensions.
    Architecture: Extends FlextTestsConstants with infra-specific constants.
    All base constants from FlextTestsConstants are available through inheritance.
    """

    class Infra(TestsFlextConstants.Infra):
        """Infra-specific test constants namespace.

        All infra-specific test constants are organized under this namespace.
        Access via c.Infra.*
        """

        class Tests:
            """Test-specific constants namespace with infra extensions.

            All test-specific constants are organized under this namespace.
            Access via c.Infra.Tests.* for infra test constants.
            """

            class Projects:
                """Project names and identifiers for infra testing."""

                FLEXT_CORE: Final[str] = "flext-core"
                FLEXT_API: Final[str] = "flext-api"
                FLEXT_CLI: Final[str] = "flext-cli"
                FLEXT_MELTANO: Final[str] = "flext-meltano"
                FLEXT_LDAP: Final[str] = "flext-ldap"
                FLEXT_LDIF: Final[str] = "flext-ldif"
                FLEXT_OBSERVABILITY: Final[str] = "flext-observability"
                FLEXT_QUALITY: Final[str] = "flext-quality"

                ALL_PROJECTS: Final[tuple[str, ...]] = (
                    FLEXT_CORE,
                    FLEXT_API,
                    FLEXT_CLI,
                    FLEXT_MELTANO,
                    FLEXT_LDAP,
                    FLEXT_LDIF,
                    FLEXT_OBSERVABILITY,
                    FLEXT_QUALITY,
                )

            class Markers:
                """Test markers for infra test categorization."""

                INFRA: Final[str] = "infra"
                INTEGRATION: Final[str] = "integration"
                DOCKER: Final[str] = "docker"
                SLOW: Final[str] = "slow"
                REQUIRES_NETWORK: Final[str] = "requires_network"
                REQUIRES_DOCKER: Final[str] = "requires_docker"

            class Versions:
                """Version strings for infra components."""

                PYTHON_MIN: Final[str] = "3.13"
                PYTHON_RECOMMENDED: Final[str] = "3.13"
                POETRY_MIN: Final[str] = "1.8"
                RUFF_MIN: Final[str] = "0.1"
                MYPY_MIN: Final[str] = "1.0"

            class Paths:
                """Path constants for infra testing."""

                DOCKER_COMPOSE_DIR: Final[str] = "docker"
                TESTS_DIR: Final[str] = "tests"
                INFRA_TESTS_DIR: Final[str] = "tests/infra"
                FIXTURES_DIR: Final[str] = "tests/fixtures"
                INTEGRATION_DIR: Final[str] = "tests/integration"


c = FlextInfraTestConstants
__all__ = ["FlextInfraTestConstants", "c"]
