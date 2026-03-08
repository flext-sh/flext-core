"""flext-core comprehensive test suite.

This package contains all tests for flext-core components.
Uses flext_tests directly for all generic test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import e

from .base import TestsFlextServiceBase
from .constants import TestsFlextConstants, c
from .models import TestsFlextModels, m
from .protocols import TestsFlextProtocols, p
from .typings import TestsFlextTypes
from .utilities import TestsFlextUtilities, u

t = TestsFlextTypes

__all__ = [
    "TestsFlextConstants",
    "TestsFlextModels",
    "TestsFlextProtocols",
    "TestsFlextServiceBase",
    "TestsFlextTypes",
    "TestsFlextUtilities",
    "c",
    "e",
    "m",
    "p",
    "t",
    "u",
]
