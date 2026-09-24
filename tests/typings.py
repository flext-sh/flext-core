"""Type system foundation for flext-core tests.

Provides TestsFlextTypes, extending TestsFlextTypes with flext-core-specific types.
All generic test types come from flext_tests, only flext-core-specific additions here.

Architecture:
- TestsFlextTypes (flext_tests) = Generic types for all FLEXT projects
- TestsFlextTypes (tests/) = flext-core-specific types extending TestsFlextTypes

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence

from flext_tests import FlextTestsTypes


class TestsFlextTypes(FlextTestsTypes):
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

        class Tests(FlextTestsTypes.Tests):
            """flext-core test types namespace."""

            type ServiceConfigMapping = FlextTestsTypes.MappingKV[
                str, FlextTestsTypes.Tests.TestobjectSerializable | MutableSequence[str]
            ]
            "Service configuration mapping specific to flext-core services."
            type HandlerConfigMapping = FlextTestsTypes.MappingKV[
                str, FlextTestsTypes.Tests.TestobjectSerializable | MutableSequence[str]
            ]
            "Handler configuration mapping specific to flext-core handlers."

            type TestCaseMap = FlextTestsTypes.MappingKV[
                str, FlextTestsTypes.Tests.TestobjectSerializable
            ]

            type InputPayloadMap = FlextTestsTypes.MappingKV[
                str, FlextTestsTypes.Tests.TestobjectSerializable
            ]

            type CentralizedUnion = str | int | float | None
            "Centralized multi-arm alias used by enforcement regression fixtures."


t = TestsFlextTypes

__all__: list[str] = ["TestsFlextTypes", "t"]
