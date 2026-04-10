"""Constants for flext-core tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests import c
from tests._constants import (
    TestsFlextCoreConstantsDomain,
    TestsFlextCoreConstantsErrors,
    TestsFlextCoreConstantsFixtures,
    TestsFlextCoreConstantsLoggings,
    TestsFlextCoreConstantsOther,
    TestsFlextCoreConstantsResult,
    TestsFlextCoreConstantsServices,
    TestsFlextCoreConstantsSettings,
    TestsFlextCoreConstantsStrings,
)


class TestsFlextCoreConstants(
    c,
):
    """Layer 0 constants facade for flext-core tests."""

    class Core:
        class Tests(
            TestsFlextCoreConstantsDomain,
            TestsFlextCoreConstantsErrors,
            TestsFlextCoreConstantsFixtures,
            TestsFlextCoreConstantsLoggings,
            TestsFlextCoreConstantsOther,
            TestsFlextCoreConstantsResult,
            TestsFlextCoreConstantsServices,
            TestsFlextCoreConstantsSettings,
            TestsFlextCoreConstantsStrings,
        ):
            pass


c = TestsFlextCoreConstants
__all__ = ["TestsFlextCoreConstants", "c"]
