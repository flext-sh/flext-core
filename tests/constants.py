"""Constants for flext-core tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests import FlextTestsConstants

from flext_core import FlextConstants
from tests import (
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
    FlextTestsConstants,
    FlextConstants,
):
    """Layer 0 constants facade for flext-core tests."""

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

__all__: list[str] = ["TestsFlextCoreConstants", "c"]
