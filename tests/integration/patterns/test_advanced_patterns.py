"""Advanced test patterns demonstrating sophisticated testing approaches.

This module showcases advanced testing patterns including:
- Given-When-Then test scenarios
- Builder patterns for test data
- Parameterized testing with complex data
- Advanced assertion patterns
- Real service patterns with state transitions

Copyright (c) 2025 FLEXT Team. All rights reserved
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    MutableSequence,
    Sequence,
)

import pytest

from tests import m, p, r, t, u

TestFunction = Callable[..., None]

pytestmark = [pytest.mark.unit, pytest.mark.architecture, pytest.mark.advanced]


def _new_config_map() -> m.ConfigMap:
    return m.ConfigMap.model_validate({})


class TestsFlextCoreAdvancedPatterns:
    """Test class demonstrating advanced testing patterns."""

    class MockScenario:
        """Mock scenario t.JsonValue for testing purposes."""

        def __init__(self, name: str, data: m.Tests.MockScenarioData) -> None:
            """Initialize mockscenario:."""
            super().__init__()
            self.name = name
            self.given = data.given
            self.when = data.when
            self.then = data.then
            self.tags = data.tags
            self.priority = data.priority

    class GivenWhenThenBuilder:
        """Builder for Given-When-Then test scenarios."""

        def __init__(self, name: str) -> None:
            """Initialize givenwhenthenbuilder:."""
            super().__init__()
            self.name = name
            self._given: m.ConfigMap = _new_config_map()
            self._when: m.ConfigMap = _new_config_map()
            self._then: m.ConfigMap = _new_config_map()
            self._tags: MutableSequence[str] = []
            self._priority = "normal"

        def given(
            self,
            _description: str,
            **kwargs: t.JsonValue,
        ) -> TestsFlextCoreAdvancedPatterns.GivenWhenThenBuilder:
            """Given method.

            Returns:
                GivenWhenThenBuilder: Self for method chaining.

            """
            self._given.update(kwargs)
            return self

        def when(
            self,
            _description: str,
            **kwargs: t.JsonValue,
        ) -> TestsFlextCoreAdvancedPatterns.GivenWhenThenBuilder:
            """When method.

            Returns:
                GivenWhenThenBuilder: Self for method chaining.

            """
            self._when.update(kwargs)
            return self

        def then(
            self,
            _description: str,
            **kwargs: t.JsonValue,
        ) -> TestsFlextCoreAdvancedPatterns.GivenWhenThenBuilder:
            """Then method.

            Returns:
                GivenWhenThenBuilder: Self for method chaining.

            """
            self._then.update(kwargs)
            return self

        def with_tag(
            self, tag: str
        ) -> TestsFlextCoreAdvancedPatterns.GivenWhenThenBuilder:
            """with_tag method.

            Returns:
                GivenWhenThenBuilder: Self for method chaining.

            """
            self._tags.append(tag)
            return self

        def with_priority(
            self,
            priority: str,
        ) -> TestsFlextCoreAdvancedPatterns.GivenWhenThenBuilder:
            """with_priority method.

            Returns:
                GivenWhenThenBuilder: Self for method chaining.

            """
            self._priority = priority
            return self

        def build(self) -> TestsFlextCoreAdvancedPatterns.MockScenario:
            """Build method.

            Returns:
                MockScenario: Built mock scenario.

            """

            def convert_dict_value(
                value: t.JsonPayload | None,
            ) -> t.JsonValue:
                """Convert t.JsonValue to t.Scalar."""
                return u.normalize_to_metadata(value)

            given_source: t.JsonMapping = {
                key: convert_dict_value(value) for key, value in self._given.items()
            }

            given_converted: t.JsonMapping = {
                str(key): convert_dict_value(value)
                for key, value in given_source.items()
            }
            when_converted: t.JsonMapping = {
                str(key): convert_dict_value(value) for key, value in self._when.items()
            }
            then_converted: t.JsonMapping = {
                str(key): convert_dict_value(value) for key, value in self._then.items()
            }
            scenario_data = m.Tests.MockScenarioData.model_validate(
                obj={
                    "given": given_converted,
                    "when": when_converted,
                    "then": then_converted,
                    "tags": self._tags,
                    "priority": self._priority,
                },
            )
            return TestsFlextCoreAdvancedPatterns.MockScenario(self.name, scenario_data)

    class FlextTestBuilder:
        """Builder for FLEXT test data with detailed metadata."""

        def __init__(self) -> None:
            """Initialize flexttestbuilder:."""
            super().__init__()
            self._data: m.ConfigMap = _new_config_map()
            self._validation_rules: m.ConfigMap = _new_config_map()

        def with_id(self, id_: str) -> TestsFlextCoreAdvancedPatterns.FlextTestBuilder:
            """with_id method.

            Returns:
                FlextTestBuilder: Self for method chaining.

            """
            self._data["id"] = id_
            return self

        def with_correlation_id(
            self,
            correlation_id: str,
        ) -> TestsFlextCoreAdvancedPatterns.FlextTestBuilder:
            """with_correlation_id method.

            Returns:
                FlextTestBuilder: Self for method chaining.

            """
            self._data["correlation_id"] = correlation_id
            return self

        def with_metadata(
            self,
            **kwargs: t.JsonValue,
        ) -> TestsFlextCoreAdvancedPatterns.FlextTestBuilder:
            """with_metadata method.

            Returns:
                FlextTestBuilder: Self for method chaining.

            """
            self._data.update(kwargs)
            return self

        def with_user_data(
            self,
            name: str,
            email: str,
        ) -> TestsFlextCoreAdvancedPatterns.FlextTestBuilder:
            """with_user_data method.

            Returns:
                FlextTestBuilder: Self for method chaining.

            """
            self._data["name"] = name
            self._data["email"] = email
            return self

        def with_timestamp(self) -> TestsFlextCoreAdvancedPatterns.FlextTestBuilder:
            """with_timestamp method.

            Returns:
                FlextTestBuilder: Self for method chaining.

            """
            self._data.setdefault("created_at", "2023-01-01T00:00:00+00:00")
            self._data.setdefault("updated_at", "2023-01-01T00:00:00+00:00")
            return self

        def with_validation_rules(
            self,
            **kwargs: t.JsonValue,
        ) -> TestsFlextCoreAdvancedPatterns.FlextTestBuilder:
            """with_validation_rules method.

            Returns:
                FlextTestBuilder: Self for method chaining.

            """
            self._validation_rules = m.ConfigMap.model_validate(kwargs)
            return self

        def build(self) -> t.JsonMapping:
            """Build method.

            Returns:
                dict: Copy of the built data.

            """
            return t.json_mapping_adapter().validate_python(self._data.model_dump())

    class ParameterizedTestBuilder:
        """Builder for parameterized test cases with success/failure scenarios."""

        def __init__(self, test_name: str) -> None:
            """Initialize parameterizedtestbuilder:."""
            super().__init__()
            self.test_name = test_name
            self._cases: MutableSequence[m.Tests.FixtureCaseDict] = []
            self._success_cases: MutableSequence[m.Tests.FixtureCaseDict] = []
            self._failure_cases: MutableSequence[m.Tests.FixtureCaseDict] = []

        def add_case(
            self,
            **kwargs: t.Scalar | MutableSequence[str],
        ) -> TestsFlextCoreAdvancedPatterns.ParameterizedTestBuilder:
            """add_case method.

            Returns:
                ParameterizedTestBuilder: Self for method chaining.

            """
            self._cases.append(m.Tests.FixtureCaseDict.model_validate(obj=kwargs))
            return self

        def add_success_cases(
            self,
            cases: Sequence[m.Tests.FixtureCaseDict],
        ) -> TestsFlextCoreAdvancedPatterns.ParameterizedTestBuilder:
            """add_success_cases method.

            Returns:
                ParameterizedTestBuilder: Self for method chaining.

            """
            self._success_cases.extend(cases)
            return self

        def add_failure_cases(
            self,
            cases: Sequence[m.Tests.FixtureCaseDict],
        ) -> TestsFlextCoreAdvancedPatterns.ParameterizedTestBuilder:
            """add_failure_cases method.

            Returns:
                ParameterizedTestBuilder: Self for method chaining.

            """
            self._failure_cases.extend(cases)
            return self

        def build(self) -> Sequence[m.Tests.FixtureCaseDict]:
            """Build method.

            Returns:
                Sequence[FixtureCaseDict]: Copy of the test cases.

            """
            return list(self._cases)

        def build_pytest_params(self) -> Sequence[tuple[str, str, bool]]:
            """build_pytest_params method.

            Returns:
                Sequence[tuple[str, str, bool]]: Pytest parameters for testing.

            """
            success_params = [
                (str(c.email), str(c.input), True) for c in self._success_cases
            ]
            failure_params = [
                (str(c.email), str(c.input), False) for c in self._failure_cases
            ]
            return success_params + failure_params

        def build_test_ids(self) -> t.StrSequence:
            """build_test_ids method.

            Returns:
                t.StrSequence: List of test IDs.

            """
            return [str(c.input) for c in (*self._success_cases, *self._failure_cases)]

    class AssertionBuilder:
        """Builder for complex test assertions."""

        def __init__(
            self,
            data: t.JsonList | t.JsonMapping | str | tuple[t.JsonValue, ...],
        ) -> None:
            """Initialize assertionbuilder:."""
            super().__init__()
            self.data: t.JsonList | t.JsonMapping | str | tuple[t.JsonValue, ...] = data
            self._assertions: MutableSequence[Callable[[], None]] = []

        def assert_equals(
            self,
            expected: Mapping[str, bool | int | str],
        ) -> TestsFlextCoreAdvancedPatterns.AssertionBuilder:
            """assert_equals method.

            Returns:
                AssertionBuilder: Self for method chaining.

            """

            def assertion() -> None:
                assert self.data == expected

            self._assertions.append(assertion)
            return self

        def assert_contains(
            self,
            item: int | str,
        ) -> TestsFlextCoreAdvancedPatterns.AssertionBuilder:
            """assert_contains method.

            Returns:
                AssertionBuilder: Self for method chaining.

            """

            def assertion() -> None:
                if isinstance(self.data, (list, tuple, set)):
                    assert item in self.data
                elif isinstance(self.data, str):
                    assert str(item) in self.data
                else:
                    assert item in self.data

            self._assertions.append(assertion)
            return self

        def assert_type(
            self,
            expected_type: type,
        ) -> TestsFlextCoreAdvancedPatterns.AssertionBuilder:
            """assert_type method.

            Returns:
                AssertionBuilder: Self for method chaining.

            """

            def assertion() -> None:
                assert isinstance(self.data, expected_type)

            self._assertions.append(assertion)
            return self

        def satisfies(
            self,
            condition: Callable[
                [
                    t.JsonList | t.JsonMapping | str | tuple[t.JsonValue, ...],
                ],
                bool,
            ],
            description: str = "",
        ) -> TestsFlextCoreAdvancedPatterns.AssertionBuilder:
            """Satisfies method.

            Args:
                condition: Function that takes the data and returns a boolean
                description: Optional description of the condition

            Returns:
                AssertionBuilder: Self for method chaining.

            """

            def assertion() -> None:
                if not condition(self.data):
                    msg = (
                        f"Condition failed: {description}"
                        if description
                        else "Condition failed"
                    )
                    raise AssertionError(msg)

            self._assertions.append(assertion)
            return self

        def execute_all(self) -> None:
            """execute_all method."""
            for assertion in self._assertions:
                assertion()

    class PatternMarker:
        """Marker for test patterns."""

        def __call__[F: Callable[..., None]](self, pattern: str) -> Callable[[F], F]:
            def decorator(func: F) -> F:
                setattr(func, "_test_pattern", pattern)
                return func

            return decorator

    mark_test_pattern = PatternMarker()

    def test_given_when_then_builder_pattern(self) -> None:
        """Test Given-When-Then builder pattern."""
        scenario: TestsFlextCoreAdvancedPatterns.MockScenario = (
            self
            .GivenWhenThenBuilder("user_registration")
            .given("user provides valid data", email="test@example.com")
            .when("registration is processed", action="register")
            .then("user is created successfully", status="success")
            .with_tag("integration")
            .with_priority("high")
            .build()
        )
        assert scenario.name == "user_registration"
        assert scenario.given["email"] == "test@example.com"
        assert scenario.when["action"] == "register"
        assert scenario.then["status"] == "success"
        assert "integration" in scenario.tags
        assert scenario.priority == "high"

    def test_flext_test_builder_pattern(self) -> None:
        """Test FLEXT test builder pattern."""
        test_data = (
            self
            .FlextTestBuilder()
            .with_id("test-123")
            .with_correlation_id("corr-456")
            .with_user_data("John Doe", "john@example.com")
            .with_timestamp()
            .with_metadata(environment="test", version="1.0")
            .build()
        )
        assert test_data.get("id") == "test-123"
        assert test_data.get("correlation_id") == "corr-456"
        assert test_data.get("name") == "John Doe"
        assert test_data.get("email") == "john@example.com"
        assert "created_at" in test_data
        assert "updated_at" in test_data
        assert test_data.get("environment") == "test"
        assert test_data.get("version") == "1.0"

    def test_parameterized_test_builder_pattern(self) -> None:
        """Test parameterized test builder pattern."""
        builder = (
            self
            .ParameterizedTestBuilder("email_validation")
            .add_success_cases([
                m.Tests.FixtureCaseDict.model_validate(
                    obj={
                        "email": "valid@example.com",
                        "input": "valid@example.com",
                    },
                ),
                m.Tests.FixtureCaseDict.model_validate(
                    obj={
                        "email": "user.name@domain.co.uk",
                        "input": "user.name@domain.co.uk",
                    },
                ),
            ])
            .add_failure_cases([
                m.Tests.FixtureCaseDict.model_validate(
                    obj={
                        "email": "invalid-email",
                        "input": "invalid-email",
                    },
                ),
                m.Tests.FixtureCaseDict.model_validate(
                    obj={
                        "email": "@domain.com",
                        "input": "@domain.com",
                    },
                ),
            ])
        )
        params = builder.build_pytest_params()
        assert len(params) == 4
        assert params[0][2] is True
        assert params[2][2] is False
        test_ids = builder.build_test_ids()
        assert len(test_ids) == 4
        assert "valid@example.com" in test_ids
        assert "invalid-email" in test_ids

    def test_assertion_builder_pattern(self) -> None:
        """Test assertion builder pattern."""
        test_data: t.JsonMapping = {
            "name": "John",
            "age": 30,
            "active": True,
        }
        assertion_builder = (
            self
            .AssertionBuilder(test_data)
            .assert_type(dict)
            .assert_contains("name")
            .assert_equals({"name": "John", "age": 30, "active": True})
        )
        assertion_builder.execute_all()

    @mark_test_pattern("mock_scenario")
    def test_mock_scenario_pattern(self) -> None:
        """Test mock scenario pattern."""
        scenario_data = m.Tests.MockScenarioData.model_validate(
            obj={
                "given": {"user": "authenticated"},
                "when": {"action": "request_data"},
                "then": {"result": "success"},
                "tags": ["api", "integration"],
                "priority": "medium",
            },
        )
        scenario = self.MockScenario("api_request", scenario_data)
        assert scenario.name == "api_request"
        assert scenario.given["user"] == "authenticated"
        assert scenario.when["action"] == "request_data"
        assert scenario.then["result"] == "success"
        assert "api" in scenario.tags
        assert scenario.priority == "medium"

    def test_advanced_service_patterns(self) -> None:
        """Test advanced service patterns with real functionality."""

        class ProcessingService:
            """Real service for testing state transitions."""

            def __init__(self) -> None:
                """Initialize processing service."""
                super().__init__()
                self.call_count = 0
                self.states = ["processing", "completed", "error"]

            def process(self, _data: str) -> p.Result[t.StrMapping]:
                """Process data with state transitions."""
                self.call_count += 1
                if self.call_count == 1:
                    return r[t.StrMapping].ok({"status": "processing"})
                if self.call_count == 2:
                    return r[t.StrMapping].ok({"status": "completed"})
                return r[t.StrMapping].fail("Service unavailable")

        service = ProcessingService()
        result1 = service.process("data1")
        assert result1.success
        assert result1.value["status"] == "processing"
        result2 = service.process("data2")
        assert result2.success
        assert result2.value["status"] == "completed"
        assert service.call_count == 2
        result3 = service.process("data3")
        assert result3.failure
        assert result3.error is not None and "Service unavailable" in result3.error

    def test_complex_test_data_generation(self) -> None:
        """Test complex test data generation."""
        scenarios: MutableSequence[TestsFlextCoreAdvancedPatterns.MockScenario] = []
        for i in range(3):
            scenario = (
                self
                .GivenWhenThenBuilder(f"scenario_{i}")
                .given(f"setup_{i}", value=i)
                .when(f"action_{i}", operation=f"op_{i}")
                .then(f"result_{i}", outcome=f"outcome_{i}")
                .build()
            )
            scenarios.append(scenario)
        assert len(scenarios) == 3
        assert scenarios[0].name == "scenario_0"
        value = scenarios[1].given.get("value")
        assert isinstance(value, int) and value == 1
        assert scenarios[2].when["operation"] == "op_2"

    def test_nested_builder_patterns(self) -> None:
        """Test nested builder patterns."""
        main_data = (
            self
            .FlextTestBuilder()
            .with_id("main-123")
            .with_metadata(
                nested_data="nested-456",
            )
            .build()
        )
        assert main_data.get("id") == "main-123"
        nested_data = main_data.get("nested_data")
        assert nested_data is not None
        assert nested_data == "nested-456"

    def test_fluent_interface_pattern(self) -> None:
        """Test fluent interface pattern."""
        result = (
            self
            .AssertionBuilder([1, 2, 3, 4, 5])
            .assert_type(list)
            .assert_contains(3)
            .satisfies(lambda x: len(x) == 5, "should have 5 elements")
        )
        result.execute_all()
        assert isinstance(result, self.AssertionBuilder)

    def test_test_pattern_marking(self) -> None:
        """Test test pattern marking functionality."""
        test_func = self.test_mock_scenario_pattern
        pattern_value = getattr(test_func, "_test_pattern", None)
        assert pattern_value == "mock_scenario"
