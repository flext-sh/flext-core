"""Constants for flext-core tests.

MRO-composed constants facade — all constants defined in _constants/ mixins.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests import FlextTestsConstants

from flext_core import FlextConstants
from tests._constants.domain import TestsFlextConstantsDomain
from tests._constants.errors import TestsFlextConstantsErrors
from tests._constants.fixtures import TestsFlextConstantsFixtures
from tests._constants.loggings import TestsFlextConstantsLoggings
from tests._constants.other import TestsFlextConstantsOther
from tests._constants.result import TestsFlextConstantsResult
from tests._constants.services import TestsFlextConstantsServices
from tests._constants.settings import TestsFlextConstantsSettings
from tests._constants.strings import TestsFlextConstantsStrings


class TestsFlextConstants(FlextTestsConstants, FlextConstants):
    """Layer 0 constants facade for flext-core tests."""

    class Tests(
        TestsFlextConstantsOther,
        TestsFlextConstantsStrings,
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
