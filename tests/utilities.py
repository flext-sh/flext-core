"""Utilities for flext-core tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests import FlextTestsUtilities

from tests._utilities.case_factories import TestsFlextUtilitiesCaseFactoriesMixin
from tests._utilities.contracts import TestsFlextUtilitiesContractsMixin
from tests._utilities.dispatch import TestsFlextUtilitiesDispatchMixin
from tests._utilities.parser_reliability import (
    TestsFlextUtilitiesParserReliabilityMixin,
)
from tests._utilities.railway import TestsFlextUtilitiesRailwayMixin
from tests._utilities.service_factories import TestsFlextUtilitiesServiceFactoriesMixin
from tests._utilities.services import TestsFlextUtilitiesServicesMixin


class TestsFlextUtilities(FlextTestsUtilities):
    """Utilities for flext-core tests."""

    class Tests(
        TestsFlextUtilitiesCaseFactoriesMixin,
        TestsFlextUtilitiesContractsMixin,
        TestsFlextUtilitiesParserReliabilityMixin,
        TestsFlextUtilitiesServiceFactoriesMixin,
        TestsFlextUtilitiesServicesMixin,
        TestsFlextUtilitiesRailwayMixin,
        TestsFlextUtilitiesDispatchMixin,
        FlextTestsUtilities.Tests,
    ):
        """flext-core test utilities namespace."""


u = TestsFlextUtilities

__all__: list[str] = ["TestsFlextUtilities", "u"]
