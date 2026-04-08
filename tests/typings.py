"""Type system foundation for flext-core tests.

Provides TestsFlextCoreTypes, extending TestsFlextTypes with flext-core-specific types.
All generic test types come from flext_tests, only flext-core-specific additions here.

Architecture:
- TestsFlextTypes (flext_tests) = Generic types for all FLEXT projects
- TestsFlextCoreTypes (tests/) = flext-core-specific types extending TestsFlextTypes

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence

from flext_cli import t
from flext_core import T, T_co, T_contra
from flext_tests import FlextTestsTypes


class TestsFlextCoreTypes(FlextTestsTypes, t):
    """Type system foundation for flext-core tests - extends TestsFlextTypes.

    Architecture: Extends TestsFlextTypes with flext-core-specific type definitions.
    All generic types from TestsFlextTypes are available through inheritance.

    Rules:
    - NEVER redeclare types from TestsFlextTypes
    - Only flext-core-specific types allowed (not generic for other projects)
    - All generic types come from TestsFlextTypes
    """

    class Core:
        """Flext-core-specific type definitions for testing.

        Uses composition of TestsFlextTypes for type safety and consistency.
        Only defines types that are truly flext-core-specific.
        """

        type ServiceConfigMapping = Mapping[
            str,
            FlextTestsTypes.ContainerValue | MutableSequence[str],
        ]
        "Service configuration mapping specific to flext-core services."
        type HandlerConfigMapping = Mapping[
            str,
            FlextTestsTypes.ContainerValue | MutableSequence[str],
        ]
        "Handler configuration mapping specific to flext-core handlers."


t = TestsFlextCoreTypes

__all__ = ["T", "T_co", "T_contra", "TestsFlextCoreTypes", "t"]
