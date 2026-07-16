"""String validation scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tests.models import m

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestsFlextUtilitiesValidationStringScenarios:
    """String validation scenarios."""

    REQUIRED_SCENARIOS: ClassVar[Sequence[p.Tests.ValidationScenario]] = [
        m.Tests.ValidationScenario(
            name="required_valid",
            validator_type="string",
            input_value="non-empty",
            should_succeed=True,
            expected_value="non-empty",
            description="Valid non-empty string",
        ),
        m.Tests.ValidationScenario(
            name="required_unicode",
            validator_type="string",
            input_value="café",
            should_succeed=True,
            expected_value="café",
            description="Unicode characters",
        ),
        m.Tests.ValidationScenario(
            name="required_special",
            validator_type="string",
            input_value="test@#$%",
            should_succeed=True,
            expected_value="test@#$%",
            description="Special characters",
        ),
        m.Tests.ValidationScenario(
            name="required_none",
            validator_type="string",
            input_value=None,
            should_succeed=False,
            expected_error_contains="empty",
            description="None value rejection",
        ),
        m.Tests.ValidationScenario(
            name="required_empty",
            validator_type="string",
            input_value="",
            should_succeed=False,
            expected_error_contains="empty",
            description="Empty string rejection",
        ),
        m.Tests.ValidationScenario(
            name="required_whitespace",
            validator_type="string",
            input_value="   ",
            should_succeed=False,
            expected_error_contains="empty",
            description="Whitespace-only rejection",
        ),
        m.Tests.ValidationScenario(
            name="required_single_char",
            validator_type="string",
            input_value="a",
            should_succeed=True,
            expected_value="a",
            description="Single character string",
        ),
    ]
    CHOICE_SCENARIOS: ClassVar[Sequence[p.Tests.ValidationScenario]] = [
        m.Tests.ValidationScenario(
            name="choice_valid_single",
            validator_type="string",
            input_value="option1",
            input_params={"valid_choices": ["option1", "option2", "option3"]},
            should_succeed=True,
            expected_value="option1",
            description="Valid single choice",
        ),
        m.Tests.ValidationScenario(
            name="choice_valid_second",
            validator_type="string",
            input_value="option2",
            input_params={"valid_choices": ["option1", "option2", "option3"]},
            should_succeed=True,
            expected_value="option2",
            description="Valid second choice",
        ),
        m.Tests.ValidationScenario(
            name="choice_invalid",
            validator_type="string",
            input_value="invalid",
            input_params={"valid_choices": ["option1", "option2"]},
            should_succeed=False,
            expected_error_contains="Must be one of",
            description="Invalid choice",
        ),
        m.Tests.ValidationScenario(
            name="choice_case_sensitive",
            validator_type="string",
            input_value="OPTION1",
            input_params={
                "valid_choices": ["option1", "option2"],
                "case_sensitive": True,
            },
            should_succeed=False,
            expected_error_contains="Must be one of",
            description="Case-sensitive choice",
        ),
        m.Tests.ValidationScenario(
            name="choice_case_insensitive",
            validator_type="string",
            input_value="option1",
            input_params={
                "valid_choices": ["option1", "option2"],
                "case_sensitive": False,
            },
            should_succeed=True,
            expected_value="option1",
            description="Case-insensitive choice",
        ),
    ]


__all__: list[str] = ["TestsFlextUtilitiesValidationStringScenarios"]
