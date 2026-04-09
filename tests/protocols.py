"""Protocol definitions for flext-core tests.

Provides TestsFlextCoreProtocols, extending TestsFlextProtocols with flext-core-specific
protocols. All generic test protocols come from flext_tests.

Architecture:
- TestsFlextProtocols (flext_tests) = Generic protocols for all FLEXT projects
- TestsFlextCoreProtocols (tests/) = flext-core-specific protocols extending TestsFlextProtocols

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextProtocols
from flext_tests import FlextTestsProtocols


class TestsFlextCoreProtocols(FlextTestsProtocols, FlextProtocols):
    """Protocol definitions for flext-core tests - extends TestsFlextProtocols.

    Architecture: Extends TestsFlextProtocols with flext-core-specific protocol
    definitions. All generic protocols from TestsFlextProtocols are available
    through inheritance.

    Rules:
    - NEVER redeclare protocols from TestsFlextProtocols
    - Only flext-core-specific protocols allowed
    - All generic protocols come from TestsFlextProtocols
    """

    class Core:
        """flext-core-specific protocol definitions namespace."""

        class Tests(FlextTestsProtocols.Tests):
            """flext-core test protocols namespace."""


p = TestsFlextCoreProtocols
__all__ = ["TestsFlextCoreProtocols", "p"]
