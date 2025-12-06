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

from flext_tests.protocols import FlextTestsProtocols


class TestsFlextProtocols(FlextTestsProtocols):
    """Protocol definitions for flext-core tests - extends FlextTestsProtocols.

    Architecture: Extends FlextTestsProtocols with flext-core-specific protocol
    definitions. All generic protocols from FlextTestsProtocols are available
    through inheritance.

    Rules:
    - NEVER redeclare protocols from FlextTestsProtocols
    - Only flext-core-specific protocols allowed
    - All generic protocols come from FlextTestsProtocols
    """

    # NOTE: FlextTestsProtocols already extends FlextProtocols.
    # All FlextProtocols and FlextTestsProtocols classes are accessible through
    # TestsFlextProtocols via inheritance.
    #
    # Available protocols include:
    # - Foundation: ResultProtocol, ResultLike, ModelProtocol
    # - Configuration: ConfigProtocol
    # - Domain: Service, Repository
    # - Application: Handler, CommandBus, Middleware
    # - Infrastructure: LoggerProtocol, Connection
    # - Docker: ContainerProtocol, DockerClientProtocol, ComposeClientProtocol, etc.
    #
    # Flext-core-specific protocols can be added here if needed.


__all__ = [
    "TestsFlextProtocols",
]
