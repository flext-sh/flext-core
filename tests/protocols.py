"""Protocol definitions for flext-core tests.

Provides TestsFlextProtocols, extending FlextTestsProtocols with flext-core-specific
protocols. All generic test protocols come from flext_tests.

Architecture:
- FlextTestsProtocols (flext_tests) = Generic protocols for all FLEXT projects
- TestsFlextProtocols (tests/) = flext-core-specific protocols extending FlextTestsProtocols

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra import FlextInfraProtocols
from flext_tests import FlextTestsProtocols


class TestsFlextProtocols(FlextTestsProtocols, FlextInfraProtocols):
    """Protocol definitions for flext-core tests - extends FlextTestsProtocols.

    Architecture: Extends FlextTestsProtocols with flext-core-specific protocol
    definitions. All generic protocols from FlextTestsProtocols are available
    through inheritance.

    Rules:
    - NEVER redeclare protocols from FlextTestsProtocols
    - Only flext-core-specific protocols allowed
    - All generic protocols come from FlextTestsProtocols
    """


p = TestsFlextProtocols
__all__ = ["TestsFlextProtocols", "p"]
