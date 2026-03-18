"""Protocol definitions for flext-core tests.

Provides TestsFlextProtocols, extending p with flext-core-specific
protocols. All generic test protocols come from flext_tests.

Architecture:
- p (flext_tests) = Generic protocols for all FLEXT projects
- TestsFlextProtocols (tests/) = flext-core-specific protocols extending p

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra import FlextInfraProtocols
from flext_tests import p


class TestsFlextProtocols(p, FlextInfraProtocols):
    """Protocol definitions for flext-core tests - extends p.

    Architecture: Extends p with flext-core-specific protocol
    definitions. All generic protocols from p are available
    through inheritance.

    Rules:
    - NEVER redeclare protocols from p
    - Only flext-core-specific protocols allowed
    - All generic protocols come from p
    """


p = TestsFlextProtocols
__all__ = ["TestsFlextProtocols", "p"]
