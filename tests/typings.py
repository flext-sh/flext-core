"""Type system foundation for flext-core tests.

Provides TestsFlextTypes, extending FlextTestsTypes with flext-core-specific types.
All generic test types come from flext_tests, only flext-core-specific additions here.

Architecture:
- FlextTestsTypes (flext_tests) = Generic types for all FLEXT projects
- TestsFlextTypes (tests/) = flext-core-specific types extending FlextTestsTypes

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from flext_infra import FlextInfraTypes
from flext_tests import FlextTestsTypes

from flext_core import T, T_co, T_contra


class TestsFlextTypes(FlextTestsTypes, FlextInfraTypes):
    """Type system foundation for flext-core tests - extends FlextTestsTypes.

    Architecture: Extends FlextTestsTypes with flext-core-specific type definitions.
    All generic types from FlextTestsTypes are available through inheritance.

    Rules:
    - NEVER redeclare types from FlextTestsTypes
    - Only flext-core-specific types allowed (not generic for other projects)
    - All generic types come from FlextTestsTypes
    """

    class Tests(FlextTestsTypes.Tests):
        """Flext-core-specific type definitions for testing.

        Uses composition of t for type safety and consistency.
        Only defines types that are truly flext-core-specific.
        """

        type ServiceConfigMapping = Mapping[
            str,
            object | Sequence[str] | Mapping[str, str | int] | None,
        ]
        "Service configuration mapping specific to flext-core services."
        type HandlerConfigMapping = Mapping[
            str,
            object | Sequence[str] | Mapping[str, str] | None,
        ]
        "Handler configuration mapping specific to flext-core handlers."


__all__ = ["T", "T_co", "T_contra", "TestsFlextTypes"]
