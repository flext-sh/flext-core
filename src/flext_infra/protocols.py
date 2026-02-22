"""Protocol definitions for flext-infra.

Defines interface contracts for infrastructure services using Python Protocol
for loose coupling and improved testability.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


class FlextInfraProtocols:
    """Namespace for infrastructure service protocols.

    Provides interface contracts for all infrastructure services including
    base.mk templating, validation, dependency management, and workspace
    orchestration.

    Usage:
        >>> from flext_infra import p
        >>> # Access protocols via p.ServiceName
    """

    pass


p = FlextInfraProtocols

__all__ = ["FlextInfraProtocols", "p"]
