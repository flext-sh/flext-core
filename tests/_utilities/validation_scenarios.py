"""Validation scenario namespace helper for flext-core tests."""

from __future__ import annotations

from tests._utilities.validation_network import (
    TestsFlextUtilitiesValidationNetworkScenarios,
)
from tests._utilities.validation_numeric import (
    TestsFlextUtilitiesValidationNumericScenarios,
)
from tests._utilities.validation_pattern import (
    TestsFlextUtilitiesValidationPatternScenarios,
)
from tests._utilities.validation_string import (
    TestsFlextUtilitiesValidationStringScenarios,
)
from tests._utilities.validation_uri import TestsFlextUtilitiesValidationUriScenarios


class TestsFlextUtilitiesValidationScenariosMixin:
    """Validation scenario namespace helper."""

    class ValidationScenarios(
        TestsFlextUtilitiesValidationUriScenarios,
        TestsFlextUtilitiesValidationNetworkScenarios,
        TestsFlextUtilitiesValidationStringScenarios,
        TestsFlextUtilitiesValidationPatternScenarios,
        TestsFlextUtilitiesValidationNumericScenarios,
    ):
        """Centralized validation scenarios - single source of truth."""


__all__: list[str] = ["TestsFlextUtilitiesValidationScenariosMixin"]
