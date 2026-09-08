"""Validation scenario namespace helper for flext-core tests."""

from __future__ import annotations

from .validation_network import TestsFlextUtilitiesValidationNetworkScenarios
from .validation_numeric import TestsFlextUtilitiesValidationNumericScenarios
from .validation_pattern import TestsFlextUtilitiesValidationPatternScenarios
from .validation_string import TestsFlextUtilitiesValidationStringScenarios
from .validation_uri import TestsFlextUtilitiesValidationUriScenarios


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
