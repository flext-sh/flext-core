"""Test utilities for FLEXT infra tests.

Provides FlextInfraTestUtilities, extending FlextTestsUtilities with infra-specific
helper functions for infrastructure testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import shutil

from flext_infra import FlextInfraUtilities
from flext_tests import FlextTestsUtilities


class FlextInfraTestUtilities(FlextTestsUtilities):
    """Infra test utilities extending FlextTestsUtilities with infra-specific helpers.

    Architecture: Extends FlextTestsUtilities with infra-specific utility functions.
    All base utilities from FlextTestsUtilities are available through inheritance.
    """

    class Infra(FlextInfraUtilities.Infra):
        """Infra-specific utilities namespace."""

        class Tests:
            """Test-specific utilities namespace with infra extensions."""

            @staticmethod
            def is_docker_available() -> bool:
                """Check if Docker is available on the system.

                Returns:
                    True if Docker is available, False otherwise.

                """
                return shutil.which("docker") is not None

            @staticmethod
            def is_project_valid(project_name: str) -> bool:
                """Check if a project name is valid for infra testing.

                Args:
                    project_name: Name of the project to validate.

                Returns:
                    True if project name is valid, False otherwise.

                """
                if not project_name:
                    return False
                return project_name.replace("-", "").replace("_", "").isalnum()


u = FlextInfraTestUtilities
__all__ = ["FlextInfraTestUtilities", "u"]
