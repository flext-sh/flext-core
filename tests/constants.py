"""Constants for flext-core tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests import FlextTestsConstants

from flext_core import FlextConstants
from tests import (
    TestsFlextConstantsDomain,
    TestsFlextConstantsErrors,
    TestsFlextConstantsFixtures,
    TestsFlextConstantsLoggings,
    TestsFlextConstantsOther,
    TestsFlextConstantsResult,
    TestsFlextConstantsServices,
    TestsFlextConstantsSettings,
    TestsFlextConstantsStrings,
)


class TestsFlextConstants(
    FlextTestsConstants,
    FlextConstants,
):
    """Layer 0 constants facade for flext-core tests."""

    class Tests(
        FlextTestsConstants.Tests,
        TestsFlextConstantsDomain,
        TestsFlextConstantsErrors,
        TestsFlextConstantsFixtures,
        TestsFlextConstantsLoggings,
        TestsFlextConstantsOther,
        TestsFlextConstantsResult,
        TestsFlextConstantsServices,
        TestsFlextConstantsSettings,
        TestsFlextConstantsStrings,
    ):
        pass


c = TestsFlextConstants

__all__: list[str] = ["TestsFlextConstants", "c"]
