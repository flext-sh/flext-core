"""Protocol definitions for FLEXT infra tests.

Provides FlextInfraTestProtocols, extending FlextTestsProtocols with infra-specific
protocol definitions for infrastructure testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tests import TestsFlextProtocols


class FlextInfraTestProtocols(TestsFlextProtocols):
    """Protocol definitions for FLEXT infra tests - extends FlextTestsProtocols.

    Architecture: Extends FlextTestsProtocols with infra-specific protocol definitions.
    All base protocols from FlextTestsProtocols are available through inheritance.
    Protocols cannot import models - only other protocols and types.
    """

    class Infra(TestsFlextProtocols.Infra):
        """Infra-specific protocol definitions."""

        class Tests:
            """Test-specific protocol definitions namespace with infra extensions."""

            @runtime_checkable
            class ProjectValidator(Protocol):
                """Protocol for project validation in infra tests."""

                def validate(self) -> bool:
                    """Validate project configuration.

                    Returns:
                        True if project is valid, False otherwise.

                    """
                    ...

            @runtime_checkable
            class InfraTestRunner(Protocol):
                """Protocol for running infra tests."""

                def run_tests(self) -> int:
                    """Run infra tests.

                    Returns:
                        Exit code (0 for success, non-zero for failure).

                    """
                    ...


p = FlextInfraTestProtocols
__all__ = ["FlextInfraTestProtocols", "p"]
