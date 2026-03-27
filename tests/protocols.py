"""Protocol definitions for flext-core tests.

Provides FlextCoreTestProtocols, extending FlextTestsProtocols with flext-core-specific
protocols. All generic test protocols come from flext_tests.

Architecture:
- FlextTestsProtocols (flext_tests) = Generic protocols for all FLEXT projects
- FlextCoreTestProtocols (tests/) = flext-core-specific protocols extending FlextTestsProtocols

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests import FlextTestsProtocols

from flext_core import FlextProtocols


class FlextCoreTestProtocols(FlextTestsProtocols, FlextProtocols):
    """Protocol definitions for flext-core tests - extends FlextTestsProtocols.

    Architecture: Extends FlextTestsProtocols with flext-core-specific protocol
    definitions. All generic protocols from FlextTestsProtocols are available
    through inheritance.

    Rules:
    - NEVER redeclare protocols from FlextTestsProtocols
    - Only flext-core-specific protocols allowed
    - All generic protocols come from FlextTestsProtocols
    """

    class Core:
        """flext-core-specific protocol definitions namespace."""


p = FlextCoreTestProtocols
__all__ = ["FlextCoreTestProtocols", "p"]
