"""Type system for FLEXT infra tests.

Provides FlextInfraTestTypes, extending FlextTestsTypes with infra-specific
type definitions for infrastructure testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra import FlextInfraTypes
from tests import TestsFlextTypes


class FlextInfraTestTypes(TestsFlextTypes):
    """Type system for FLEXT infra tests - extends FlextTestsTypes.

    Architecture: Extends FlextTestsTypes with infra-specific type definitions.
    All base types from FlextTestsTypes are available through inheritance.
    """

    class Infra(FlextInfraTypes.Infra):
        """Infra-specific type definitions namespace."""

        class Tests:
            """Test-specific type definitions namespace with infra extensions."""

            type ProjectName = str
            "Type for project names in infra testing."
            type DockerImageName = str
            "Type for Docker image names."
            type TestMarker = str
            "Type for test markers (infra, integration, docker, etc.)."


t = FlextInfraTestTypes
__all__ = ["FlextInfraTestTypes", "t"]
