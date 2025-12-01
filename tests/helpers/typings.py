"""Type system foundation for flext-core tests.

Provides TestTypings, extending FlextTestsTypings with flext-core-specific types.
All generic test types come from flext_tests, only flext-core-specific additions here.

Architecture:
- FlextTestsTypings (flext_tests) = Generic types for all FLEXT projects
- TestTypings (tests/helpers) = flext-core-specific types that extend FlextTestsTypings

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from flext_core.typings import FlextTypes
from flext_tests.typings import (
    FlextTestsTypings,
    T,
    T_co,
    T_contra,
    TTestModel,
    TTestResult,
    TTestService,
)


class TestTypings(FlextTestsTypings):
    """Type system foundation for flext-core tests - extends FlextTestsTypings.

    Architecture: Extends FlextTestsTypings with flext-core-specific type definitions.
    All generic types from FlextTestsTypings are available through inheritance.

    Rules:
    - NEVER redeclare types from FlextTestsTypings
    - Only flext-core-specific types allowed (not generic for other projects)
    - All generic types come from FlextTestsTypings
    """

    # Flext-core-specific type additions (if any)
    # All generic types are inherited from FlextTestsTypings
    class Core:
        """Flext-core-specific type definitions for testing (not generic for other FLEXT projects).

        Uses composition of FlextTypes for type safety and consistency.
        Only defines types that are truly flext-core-specific and not generic.
        """

        # Service configuration - more restrictive than general ConfigurationMapping
        # Uses composition of FlextTypes.GeneralValueType for consistency
        type ServiceConfigMapping = Mapping[
            str,
            FlextTypes.GeneralValueType
            | Sequence[str]
            | Mapping[str, str | int]
            | None,
        ]
        """Service configuration mapping specific to flext-core services.

        More restrictive than FlextTypes.Types.ConfigurationMapping,
        allowing only specific value types for service configuration.
        """

        # Handler configuration - more restrictive than general ConfigurationMapping
        # Uses composition of FlextTypes.GeneralValueType for consistency
        type HandlerConfigMapping = Mapping[
            str,
            FlextTypes.GeneralValueType | Sequence[str] | Mapping[str, str] | None,
        ]
        """Handler configuration mapping specific to flext-core handlers.

        More restrictive than FlextTypes.Types.ConfigurationMapping,
        allowing only specific value types for handler configuration.
        """


__all__ = [
    "T",
    "TTestModel",
    "TTestResult",
    "TTestService",
    "T_co",
    "T_contra",
    "TestTypings",
]
