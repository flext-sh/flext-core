"""Pattern validation scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tests import m, p

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestsFlextUtilitiesValidationPatternScenarios:
    """Pattern validation scenarios."""

    LENGTH_SCENARIOS: ClassVar[Sequence[p.Tests.ValidationScenario]] = [
        m.Tests.ValidationScenario(
            name="length_exact",
            validator_type="string",
            input_value="12345",
            input_params={"min_length": 5, "max_length": 5},
            should_succeed=True,
            expected_value="12345",
            description="Exact length match",
        ),
        m.Tests.ValidationScenario(
            name="length_within_bounds",
            validator_type="string",
            input_value="hello",
            input_params={"min_length": 3, "max_length": 10},
            should_succeed=True,
            expected_value="hello",
            description="Length within bounds",
        ),
        m.Tests.ValidationScenario(
            name="length_below_min",
            validator_type="string",
            input_value="hi",
            input_params={"min_length": 3},
            should_succeed=False,
            expected_error_contains="at least",
            description="Length below minimum",
        ),
        m.Tests.ValidationScenario(
            name="length_above_max",
            validator_type="string",
            input_value="toolongstring",
            input_params={"max_length": 5},
            should_succeed=False,
            expected_error_contains="no more than",
            description="Length above maximum",
        ),
        m.Tests.ValidationScenario(
            name="length_zero_max",
            validator_type="string",
            input_value="",
            input_params={"min_length": 0, "max_length": 0},
            should_succeed=True,
            expected_value="",
            description="Zero-length string allowed",
        ),
    ]
    PATTERN_SCENARIOS: ClassVar[Sequence[p.Tests.ValidationScenario]] = [
        m.Tests.ValidationScenario(
            name="pattern_email_valid",
            validator_type="string",
            input_value="test@example.com",
            input_params={"pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"},
            should_succeed=True,
            expected_value="test@example.com",
            description="Valid email pattern",
        ),
        m.Tests.ValidationScenario(
            name="pattern_digits_only",
            validator_type="string",
            input_value="12345",
            input_params={"pattern": "^\\d+$"},
            should_succeed=True,
            expected_value="12345",
            description="Digits-only pattern",
        ),
        m.Tests.ValidationScenario(
            name="pattern_alphanumeric",
            validator_type="string",
            input_value="abc123",
            input_params={"pattern": "^[a-zA-Z0-9]+$"},
            should_succeed=True,
            expected_value="abc123",
            description="Alphanumeric pattern",
        ),
        m.Tests.ValidationScenario(
            name="pattern_mismatch",
            validator_type="string",
            input_value="invalid@",
            input_params={"pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"},
            should_succeed=False,
            expected_error_contains="format is invalid",
            description="Pattern mismatch",
        ),
    ]


__all__: list[str] = ["TestsFlextUtilitiesValidationPatternScenarios"]
