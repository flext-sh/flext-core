"""Models for FLEXT infra tests.

Provides FlextInfraTestModels, extending FlextTestsModels with infra-specific
model definitions for infrastructure testing and validation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from flext_infra import FlextInfraModels
from flext_tests import FlextTestsModels


class FlextInfraTestModels(FlextTestsModels):
    """Infra test models extending FlextTestsModels with infra-specific models.

    Architecture: Extends FlextTestsModels with infra-specific model definitions.
    All base models from FlextTestsModels are available through inheritance.
    """

    class Infra(FlextInfraModels.Infra):
        """Infra-specific models namespace."""

        class Tests:
            """Test-specific models namespace with infra extensions."""

            class ProjectInfo(FlextTestsModels.Value):
                """Project information model for infra testing."""

                name: str
                path: str
                version: str = "0.1.0"
                is_active: bool = True

            class InfraConfig(FlextTestsModels.Value):
                """Infrastructure configuration model."""

                project_name: str
                docker_enabled: bool = False
                test_timeout: int = 60
                parallel_tests: bool = True

            class TestResult(FlextTestsModels.Value):
                """Test result model for infra tests."""

                test_name: str
                passed: bool
                duration_ms: float = 0.0
                error_message: str | None = None


m = FlextInfraTestModels
__all__ = ["FlextInfraTestModels", "m"]
