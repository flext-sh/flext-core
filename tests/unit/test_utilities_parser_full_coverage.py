"""Behavior-focused tests for the public parser API.

This file intentionally targets only `u.parse`, because that is the parser
surface with real non-test consumers inside flext-core. Internal helpers and
normalization helpers are not asserted here so coverage reflects public usage.
"""

from __future__ import annotations

import pytest

from flext_tests import tm
from tests import c, m, u


class TestUtilitiesParserFullCoverage:
    @pytest.mark.parametrize(
        "scenario",
        [
            scenario
            for scenario in u.Core.Tests.ParserScenarios.PUBLIC_PARSE_CASES
            if scenario.should_succeed and scenario.expected_value is not None
        ],
        ids=lambda scenario: scenario.name,
    )
    def test_parse_primitives_with_public_defaults(
        self,
        scenario: m.Core.Tests.PublicParseCase,
    ) -> None:
        result = (
            u.parse(scenario.input_value, scenario.target, options=scenario.options)
            if scenario.options is not None
            else u.parse(scenario.input_value, scenario.target)
        )
        tm.ok(result, eq=scenario.expected_value)

    @pytest.mark.parametrize(
        "scenario",
        [
            scenario
            for scenario in u.Core.Tests.ParserScenarios.PUBLIC_PARSE_CASES
            if scenario.should_succeed and scenario.target is c.Core.Tests.StatusEnum
        ],
        ids=lambda scenario: scenario.name,
    )
    def test_parse_enum_values_through_public_api(
        self,
        scenario: m.Core.Tests.PublicParseCase,
    ) -> None:
        result = (
            u.parse(scenario.input_value, scenario.target, options=scenario.options)
            if scenario.options is not None
            else u.parse(scenario.input_value, scenario.target)
        )
        tm.ok(result, eq=scenario.expected_value)

    @pytest.mark.parametrize(
        "scenario",
        [
            scenario
            for scenario in u.Core.Tests.ParserScenarios.PUBLIC_PARSE_CASES
            if scenario.should_succeed and scenario.expected_data is not None
        ],
        ids=lambda scenario: scenario.name,
    )
    def test_parse_model_payload_through_public_api(
        self,
        scenario: m.Core.Tests.PublicParseCase,
    ) -> None:
        result = u.parse(scenario.input_value, scenario.target)
        payload = tm.ok(result)
        tm.that(payload, is_=scenario.target)
        tm.that(payload.model_dump(), eq=scenario.expected_data)

    @pytest.mark.parametrize(
        "scenario",
        [
            scenario
            for scenario in u.Core.Tests.ParserScenarios.PUBLIC_PARSE_CASES
            if not scenario.should_succeed
        ],
        ids=lambda scenario: scenario.name,
    )
    def test_parse_invalid_inputs_fail_via_public_result_contract(
        self,
        scenario: m.Core.Tests.PublicParseCase,
    ) -> None:
        result = (
            u.parse(scenario.input_value, scenario.target, options=scenario.options)
            if scenario.options is not None
            else u.parse(scenario.input_value, scenario.target)
        )
        error = tm.fail(result)
        if scenario.error_contains is not None:
            tm.that(error, has=scenario.error_contains)
