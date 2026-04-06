"""Type system foundation for flext-core tests.

Provides FlextCoreTestTypes, extending FlextTestsTypes with flext-core-specific types.
All generic test types come from flext_tests, only flext-core-specific additions here.

Architecture:
- FlextTestsTypes (flext_tests) = Generic types for all FLEXT projects
- FlextCoreTestTypes (tests/) = flext-core-specific types extending FlextTestsTypes

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence

from flext_cli import FlextCliTypes
from flext_core import T, T_co, T_contra
from flext_tests import FlextTestsTypes


class FlextCoreTestTypes(FlextTestsTypes, FlextCliTypes):
    """Type system foundation for flext-core tests - extends FlextTestsTypes.

    Architecture: Extends FlextTestsTypes with flext-core-specific type definitions.
    All generic types from FlextTestsTypes are available through inheritance.

    Rules:
    - NEVER redeclare types from FlextTestsTypes
    - Only flext-core-specific types allowed (not generic for other projects)
    - All generic types come from FlextTestsTypes
    """

    class Core:
        """Flext-core-specific type definitions for testing.

        Uses composition of FlextTestsTypes for type safety and consistency.
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


t = FlextCoreTestTypes

__all__ = ["FlextCoreTestTypes", "T", "T_co", "T_contra", "t"]
