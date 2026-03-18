"""Type system foundation for flext-core tests.

Provides TestsFlextTypes, extending t with flext-core-specific types.
All generic test types come from flext_tests, only flext-core-specific additions here.

Architecture:
- t (flext_tests) = Generic types for all FLEXT projects
- TestsFlextTypes (tests/) = flext-core-specific types extending t

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from flext_infra import FlextInfraTypes
from flext_tests import t

from flext_core import T, T_co, T_contra, t as ft


class TestsFlextTypes(t, FlextInfraTypes):
    """Type system foundation for flext-core tests - extends t.

    Architecture: Extends t with flext-core-specific type definitions.
    All generic types from t are available through inheritance.

    Rules:
    - NEVER redeclare types from t
    - Only flext-core-specific types allowed (not generic for other projects)
    - All generic types come from t
    """

    class Tests(t.Tests):
        """Flext-core-specific type definitions for testing.

        Uses composition of t for type safety and consistency.
        Only defines types that are truly flext-core-specific.
        """

        type ServiceConfigMapping = Mapping[str, ft.NormalizedValue | Sequence[str]]
        "Service configuration mapping specific to flext-core services."
        type HandlerConfigMapping = Mapping[str, ft.NormalizedValue | Sequence[str]]
        "Handler configuration mapping specific to flext-core handlers."


t = TestsFlextTypes

__all__ = ["T", "T_co", "T_contra", "TestsFlextTypes", "t"]
