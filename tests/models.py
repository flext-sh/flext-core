"""Models for flext-core tests.

Provides TestsFlextCoreModels using composition with TestsFlextModels and TestsFlextModels.
All generic test models come from flext_tests.

Architecture:
- TestsFlextModels (flext_tests) = Generic models for all FLEXT projects
- TestsFlextModels (flext_core) = Core domain models
- TestsFlextCoreModels (tests/) = flext-core-specific models using composition

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests import m

from tests._models.mixins import TestsFlextCoreModelsMixins


class TestsFlextCoreModels(m):
    """Models for flext-core tests - uses composition with TestsFlextModels.

    Architecture: Uses composition (not inheritance) with TestsFlextModels and TestsFlextModels
    for flext-core-specific model definitions.

    Access patterns:
    - TestsFlextCoreModels.Tests.* = flext_tests test models (via inheritance)
    - TestsFlextCoreModels.Core.Tests.* = flext-core-specific test models
    - TestsFlextCoreModels.Entity, .Value, etc. = TestsFlextModels domain models (via inheritance)

    Rules:
    - flext-core-specific models go in Core namespace
    - Generic models accessed via TestsFlextModels.Tests namespace
    """

    class Core:
        """flext-core-specific test models namespace."""

        class Tests(TestsFlextCoreModelsMixins):
            """flext-core test models namespace."""


m = TestsFlextCoreModels

__all__: list[str] = [
    "TestsFlextCoreModels",
    "m",
]
