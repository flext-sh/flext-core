"""Constants for flext-core tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests import c
from tests._constants.domain import TestsFlextCoreConstantsDomain
from tests._constants.errors import TestsFlextCoreConstantsErrors
from tests._constants.fixtures import TestsFlextCoreConstantsFixtures
from tests._constants.loggings import TestsFlextCoreConstantsLoggings
from tests._constants.other import TestsFlextCoreConstantsOther
from tests._constants.result import TestsFlextCoreConstantsResult
from tests._constants.services import TestsFlextCoreConstantsServices
from tests._constants.settings import TestsFlextCoreConstantsSettings
from tests._constants.strings import TestsFlextCoreConstantsStrings


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
