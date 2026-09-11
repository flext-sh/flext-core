"""Constants for flext-core tests.

MRO-composed constants facade — all constants defined in _constants/ mixins.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests import FlextTestsConstants

from flext_core import FlextConstants

from ._constants import (
    TestsFlextConstantsDomain,
    TestsFlextConstantsErrors,
    TestsFlextConstantsFixtures,
    TestsFlextConstantsLoggings,
    TestsFlextConstantsOther,
    TestsFlextConstantsResult,
    TestsFlextConstantsServices,
    TestsFlextConstantsSettings,
)


class TestsFlextConstants(FlextTestsConstants, FlextConstants):
    """Layer 0 constants facade for flext-core tests."""

    class Tests(
        TestsFlextConstantsOther,
        TestsFlextConstantsResult,
        TestsFlextConstantsSettings,
        TestsFlextConstantsLoggings,
        TestsFlextConstantsFixtures,
        TestsFlextConstantsServices,
        TestsFlextConstantsErrors,
        TestsFlextConstantsDomain,
        FlextTestsConstants.Tests,
    ):
        """Flat constants for test composition and parametrization.

        All constants come from MRO-composed mixins in _constants/.
        """


c = TestsFlextConstants

__all__: list[str] = ["TestsFlextConstants", "c"]
