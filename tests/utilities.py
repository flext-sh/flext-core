"""Utilities for flext-core tests."""

from __future__ import annotations

from flext_tests import u as tests_u
from tests._utilities.case_factories import TestsFlextUtilitiesCaseFactoriesMixin
from tests._utilities.contracts import TestsFlextUtilitiesContractsMixin
from tests._utilities.dispatch import TestsFlextUtilitiesDispatchMixin
from tests._utilities.parser_reliability import (
    TestsFlextUtilitiesParserReliabilityMixin,
)
from tests._utilities.railway import TestsFlextUtilitiesRailwayMixin
from tests._utilities.service_factories import TestsFlextUtilitiesServiceFactoriesMixin
from tests._utilities.services import TestsFlextUtilitiesServicesMixin
from tests._utilities.validation_scenarios import (
    TestsFlextUtilitiesValidationScenariosMixin,
)


class TestsFlextUtilities(tests_u):
    """Utilities for flext-core tests."""

    class Tests(
        TestsFlextUtilitiesCaseFactoriesMixin,
        TestsFlextUtilitiesContractsMixin,
        TestsFlextUtilitiesParserReliabilityMixin,
        TestsFlextUtilitiesServiceFactoriesMixin,
        TestsFlextUtilitiesServicesMixin,
        TestsFlextUtilitiesValidationScenariosMixin,
        TestsFlextUtilitiesRailwayMixin,
        TestsFlextUtilitiesDispatchMixin,
        tests_u.Tests,
    ):
        """flext-core test utilities namespace."""


u = TestsFlextUtilities

__all__: list[str] = ["TestsFlextUtilities", "u"]
