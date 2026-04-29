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

from collections.abc import (
    Mapping,
    MutableSequence,
)

from flext_tests import t


class TestsFlextTypes(t):
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

        class Tests(t.Tests):
            """flext-core test types namespace."""

            type ServiceConfigMapping = Mapping[
                str,
                t.Tests.TestobjectSerializable | MutableSequence[str],
            ]
            "Service configuration mapping specific to flext-core services."
            type HandlerConfigMapping = Mapping[
                str,
                t.Tests.TestobjectSerializable | MutableSequence[str],
            ]
            "Handler configuration mapping specific to flext-core handlers."

            type TestCaseMap = Mapping[str, t.Tests.TestobjectSerializable]

            type InputPayloadMap = Mapping[str, t.Tests.TestobjectSerializable]


t = TestsFlextTypes

__all__: list[str] = ["TestsFlextTypes", "t"]
