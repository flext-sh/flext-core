"""Models for flext-core tests.

Provides TestsFlextModels using composition with TestsFlextModels and TestsFlextModels.
All generic test models come from flext_tests.

Architecture:
- TestsFlextModels (flext_tests) = Generic models for all FLEXT projects
- TestsFlextModels (flext_core) = Core domain models
- TestsFlextModels (tests/) = flext-core-specific models using composition

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from flext_tests import m

from tests._models.mixins import TestsFlextModelsMixins


class TestsFlextModels(m):
    """Models for flext-core tests - uses composition with TestsFlextModels.

    Architecture: Uses composition (not inheritance) with TestsFlextModels and TestsFlextModels
    for flext-core-specific model definitions.

    Access patterns:
    - TestsFlextModels.Tests.* = flext_tests test models (via inheritance)
    - TestsFlextModels.Tests.* = flext-core-specific test models
    - TestsFlextModels.Entity, .Value, etc. = TestsFlextModels domain models (via inheritance)

    Rules:
    - flext-core-specific models go in Core namespace
    - Generic models accessed via TestsFlextModels.Tests namespace
    """

    @override
    class Tests(TestsFlextModelsMixins):
        """flext-core test models namespace."""


m = TestsFlextModels

__all__: list[str] = [
    "TestsFlextModels",
    "m",
]
