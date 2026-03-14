"""Models for FLEXT infra tests.

Provides FlextInfraTestModels, extending TestsFlextModels with infra-specific
model definitions for infrastructure testing and validation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests import TestsFlextModels


class FlextInfraTestModels(TestsFlextModels):
    """Infra test models extending TestsFlextModels with infra-specific models.

    Architecture: Extends TestsFlextModels with infra-specific model definitions.
    All base models from TestsFlextModels are available through inheritance.
    """

    class Infra(TestsFlextModels.Infra):
        """Infra-specific models namespace."""

        class Tests:
            """Test-specific models namespace with infra extensions."""

            class ProjectInfo(TestsFlextModels.Value):
                """Project information model for infra testing."""

                name: str
                path: str
                version: str = "0.1.0"
                is_active: bool = True

            class InfraConfig(TestsFlextModels.Value):
                """Infrastructure configuration model."""

                project_name: str
                docker_enabled: bool = False
                test_timeout: int = 60
                parallel_tests: bool = True

            class TestResult(TestsFlextModels.Value):
                """Test result model for infra tests."""

                test_name: str
                passed: bool
                duration_ms: float = 0.0
                error_message: str | None = None


m = FlextInfraTestModels

__all__ = ["FlextInfraTestModels", "m"]
