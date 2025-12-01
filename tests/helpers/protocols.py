"""Protocol definitions for flext-core tests.

Provides TestProtocols, extending FlextTestProtocols with flext-core-specific protocols.
All generic test protocols come from flext_tests, only flext-core-specific additions here.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests.protocols import FlextTestProtocols


class TestProtocols(FlextTestProtocols):
    """Protocol definitions for flext-core tests - extends FlextTestProtocols.

    Architecture: Extends FlextTestProtocols with flext-core-specific protocol definitions.
    All generic protocols from FlextTestProtocols are available through inheritance.

    Rules:
    - NEVER redeclare protocols from FlextTestProtocols
    - Only flext-core-specific protocols allowed
    - All generic protocols come from FlextTestProtocols
    """

    # Flext-core-specific protocol additions (if any)
    # All generic protocols (including Docker protocols) are inherited from FlextTestProtocols
    # Add flext-core-specific protocols here if needed


__all__ = ["TestProtocols"]
