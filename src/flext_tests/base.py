"""Service base for FLEXT tests.

Provides FlextTestsServiceBase, extending FlextService with test-specific service
functionality for test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.service import FlextService
from flext_core.typings import T


class FlextTestsServiceBase(FlextService[T]):
    """Service base for FLEXT tests - extends FlextService.

    Architecture: Extends FlextService with test-specific service functionality.
    All base service functionality from FlextService is available through inheritance.
    """


__all__ = ["FlextTestsServiceBase", "s"]

# Alias for simplified usage
s = FlextTestsServiceBase
