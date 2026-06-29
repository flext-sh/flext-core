"""Numeric validation scenarios."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from tests import m


class TestsFlextUtilitiesValidationNumericScenarios:
    """Numeric validation scenarios."""

    NON_NEGATIVE_SCENARIOS: ClassVar[Sequence[m.Tests.ValidationScenario]] = [
        m.Tests.ValidationScenario(
            name="non_negative_zero",
            validator_type="numeric",
            input_value=0,
            should_succeed=True,
            expected_value=0,
            description="Zero is non-negative",
        ),
        m.Tests.ValidationScenario(
            name="non_negative_positive",
            validator_type="numeric",
            input_value=42,
            should_succeed=True,
            expected_value=42,
            description="Positive number",
        ),
        m.Tests.ValidationScenario(
            name="non_negative_large",
            validator_type="numeric",
            input_value=1000000,
            should_succeed=True,
            expected_value=1000000,
            description="Large positive number",
        ),
        m.Tests.ValidationScenario(
            name="non_negative_negative",
            validator_type="numeric",
            input_value=-1,
            should_succeed=False,
            expected_error_contains="non-negative",
            description="Negative rejection",
        ),
        m.Tests.ValidationScenario(
            name="non_negative_none",
            validator_type="numeric",
            input_value=None,
            should_succeed=False,
            expected_error_contains="cannot be None",
            description="None rejection",
        ),
    ]
    POSITIVE_SCENARIOS: ClassVar[Sequence[m.Tests.ValidationScenario]] = [
        m.Tests.ValidationScenario(
            name="positive_one",
            validator_type="numeric",
            input_value=1,
            should_succeed=True,
            expected_value=1,
            description="Positive value 1",
        ),
        m.Tests.ValidationScenario(
            name="positive_large",
            validator_type="numeric",
            input_value=999999,
            should_succeed=True,
            expected_value=999999,
            description="Large positive",
        ),
        m.Tests.ValidationScenario(
            name="positive_float",
            validator_type="numeric",
            input_value=0.1,
            should_succeed=True,
            expected_value=0.1,
            description="Positive float",
        ),
        m.Tests.ValidationScenario(
            name="positive_zero",
            validator_type="numeric",
            input_value=0,
            should_succeed=False,
            expected_error_contains="positive",
            description="Zero rejection",
        ),
        m.Tests.ValidationScenario(
            name="positive_negative",
            validator_type="numeric",
            input_value=-5,
            should_succeed=False,
            expected_error_contains="positive",
            description="Negative rejection",
        ),
        m.Tests.ValidationScenario(
            name="positive_none",
            validator_type="numeric",
            input_value=None,
            should_succeed=False,
            expected_error_contains="cannot be None",
            description="None rejection",
        ),
    ]
    RANGE_SCENARIOS: ClassVar[Sequence[m.Tests.ValidationScenario]] = [
        m.Tests.ValidationScenario(
            name="range_within_bounds",
            validator_type="numeric",
            input_value=5,
            input_params={"min_value": 1, "max_value": 10},
            should_succeed=True,
            expected_value=5,
            description="Value within range",
        ),
        m.Tests.ValidationScenario(
            name="range_at_min",
            validator_type="numeric",
            input_value=1,
            input_params={"min_value": 1, "max_value": 10},
            should_succeed=True,
            expected_value=1,
            description="Value at minimum",
        ),
        m.Tests.ValidationScenario(
            name="range_at_max",
            validator_type="numeric",
            input_value=10,
            input_params={"min_value": 1, "max_value": 10},
            should_succeed=True,
            expected_value=10,
            description="Value at maximum",
        ),
        m.Tests.ValidationScenario(
            name="range_below_min",
            validator_type="numeric",
            input_value=0,
            input_params={"min_value": 1, "max_value": 10},
            should_succeed=False,
            expected_error_contains="at least",
            description="Value below minimum",
        ),
        m.Tests.ValidationScenario(
            name="range_above_max",
            validator_type="numeric",
            input_value=11,
            input_params={"min_value": 1, "max_value": 10},
            should_succeed=False,
            expected_error_contains="at most",
            description="Value above maximum",
        ),
        m.Tests.ValidationScenario(
            name="range_negative_range",
            validator_type="numeric",
            input_value=-5,
            input_params={"min_value": -10, "max_value": -1},
            should_succeed=True,
            expected_value=-5,
            description="Negative range",
        ),
        m.Tests.ValidationScenario(
            name="range_fractional",
            validator_type="numeric",
            input_value=2.5,
            input_params={"min_value": 0.5, "max_value": 5.5},
            should_succeed=True,
            expected_value=2.5,
            description="Fractional range",
        ),
        m.Tests.ValidationScenario(
            name="range_single_value",
            validator_type="numeric",
            input_value=5,
            input_params={"min_value": 5, "max_value": 5},
            should_succeed=True,
            expected_value=5,
            description="Single value range",
        ),
    ]


__all__: list[str] = ["TestsFlextUtilitiesValidationNumericScenarios"]
