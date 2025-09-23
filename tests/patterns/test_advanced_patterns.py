"""Advanced test patterns demonstrating sophisticated testing approaches.

This module showcases advanced testing patterns including:
- Given-When-Then test scenarios
- Builder patterns for test data
- Parameterized testing with complex data
- Advanced assertion patterns
- Mock scenarios and test builders

Copyright (c) 2025 FLEXT Team. All rights reserved
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from unittest.mock import Mock

import pytest

from flext_core import FlextTypes

# Type alias for test functions
TestFunction = Callable[[object], None]


def mark_test_pattern(pattern: str) -> Callable[[object], object]:
    """Mark test with a specific pattern for demonstration purposes.

    Returns:
        Callable[[object], object]: Decorator function that marks tests with patterns.

    """

    def decorator(func: object) -> object:
        """Decorator method.

        Returns:
            object: The decorated function with pattern attribute.

        """
        # Use setattr to dynamically assign the pattern attribute
        setattr(func, "_test_pattern", pattern)
        return func

    return decorator


pytestmark = [pytest.mark.unit, pytest.mark.architecture, pytest.mark.advanced]


class MockScenario:
    """Mock scenario object for testing purposes."""

    def __init__(self, name: str, data: FlextTypes.Core.Dict) -> None:
        """Initialize mockscenario:."""
        self.name = name
        self.given = cast("FlextTypes.Core.Dict", data.get("given", {}))
        self.when = cast("FlextTypes.Core.Dict", data.get("when", {}))
        self.then = cast("FlextTypes.Core.Dict", data.get("then", {}))
        self.tags = cast("FlextTypes.Core.StringList", data.get("tags", []))
        self.priority = str(data.get("priority", "normal"))


class GivenWhenThenBuilder:
    """Builder for Given-When-Then test scenarios."""

    def __init__(self, name: str) -> None:
        """Initialize givenwhenthenbuilder:."""
        self.name = name
        self._given: FlextTypes.Core.Dict = {}
        self._when: FlextTypes.Core.Dict = {}
        self._then: FlextTypes.Core.Dict = {}
        self._tags: FlextTypes.Core.StringList = []
        self._priority = "normal"

    def given(self, _description: str, **kwargs: object) -> GivenWhenThenBuilder:
        """Given method.

        Returns:
            GivenWhenThenBuilder: Self for method chaining.

        """
        self._given.update(kwargs)
        return self

    def when(self, _description: str, **kwargs: object) -> GivenWhenThenBuilder:
        """When method.

        Returns:
            GivenWhenThenBuilder: Self for method chaining.

        """
        self._when.update(kwargs)
        return self

    def then(self, _description: str, **kwargs: object) -> GivenWhenThenBuilder:
        """Then method.

        Returns:
            GivenWhenThenBuilder: Self for method chaining.

        """
        self._then.update(kwargs)
        return self

    def with_tag(self, tag: str) -> GivenWhenThenBuilder:
        """with_tag method.

        Returns:
            GivenWhenThenBuilder: Self for method chaining.

        """
        self._tags.append(tag)
        return self

    def with_priority(self, priority: str) -> GivenWhenThenBuilder:
        """with_priority method.

        Returns:
            GivenWhenThenBuilder: Self for method chaining.

        """
        self._priority = priority
        return self

    def build(self) -> MockScenario:
        """Build method.

        Returns:
            MockScenario: Built mock scenario.

        """
        return MockScenario(
            self.name,
            {
                "given": self._given,
                "when": self._when,
                "then": self._then,
                "tags": self._tags,
                "priority": self._priority,
            },
        )


class FlextTestBuilder:
    """Builder for FLEXT test data with detailed metadata."""

    def __init__(self) -> None:
        """Initialize flexttestbuilder:."""
        self._data: FlextTypes.Core.Dict = {}

    def with_id(self, id_: str) -> FlextTestBuilder:
        """with_id method.

        Returns:
            FlextTestBuilder: Self for method chaining.

        """
        self._data["id"] = id_
        return self

    def with_correlation_id(self, correlation_id: str) -> FlextTestBuilder:
        """with_correlation_id method.

        Returns:
            FlextTestBuilder: Self for method chaining.

        """
        self._data["correlation_id"] = correlation_id
        return self

    def with_metadata(self, **kwargs: object) -> FlextTestBuilder:
        """with_metadata method.

        Returns:
            FlextTestBuilder: Self for method chaining.

        """
        self._data.update(kwargs)
        return self

    def with_user_data(self, name: str, email: str) -> FlextTestBuilder:
        """with_user_data method.

        Returns:
            FlextTestBuilder: Self for method chaining.

        """
        self._data["name"] = name
        self._data["email"] = email
        return self

    def with_timestamp(self) -> FlextTestBuilder:
        """with_timestamp method.

        Returns:
            FlextTestBuilder: Self for method chaining.

        """
        self._data.setdefault("created_at", "2023-01-01T00:00:00+00:00")
        self._data.setdefault("updated_at", "2023-01-01T00:00:00+00:00")
        return self

    def with_validation_rules(self, **kwargs: object) -> FlextTestBuilder:
        """with_validation_rules method.

        Returns:
            FlextTestBuilder: Self for method chaining.

        """
        # No-op stub to keep example API; could attach schema metadata here
        # Store kwargs for potential future use
        self._validation_rules = kwargs
        return self

    def build(self) -> FlextTypes.Core.Dict:
        """Build method.

        Returns:
            FlextTypes.Core.Dict: Copy of the built data.

        """
        return self._data.copy()


class ParameterizedTestBuilder:
    """Builder for parameterized test cases with success/failure scenarios."""

    def __init__(self, test_name: str) -> None:
        """Initialize parameterizedtestbuilder:."""
        self.test_name = test_name
        self._cases: list[FlextTypes.Core.Dict] = []
        self._success_cases: list[FlextTypes.Core.Dict] = []
        self._failure_cases: list[FlextTypes.Core.Dict] = []

    def add_case(self, **kwargs: object) -> ParameterizedTestBuilder:
        """add_case method.

        Returns:
            ParameterizedTestBuilder: Self for method chaining.

        """
        self._cases.append(kwargs)
        return self

    def add_success_cases(
        self,
        cases: list[FlextTypes.Core.Dict],
    ) -> ParameterizedTestBuilder:
        """add_success_cases method.

        Returns:
            ParameterizedTestBuilder: Self for method chaining.

        """
        self._success_cases.extend(cases)
        return self

    def add_failure_cases(
        self,
        cases: list[FlextTypes.Core.Dict],
    ) -> ParameterizedTestBuilder:
        """add_failure_cases method.

        Returns:
            ParameterizedTestBuilder: Self for method chaining.

        """
        self._failure_cases.extend(cases)
        return self

    def build(self) -> list[FlextTypes.Core.Dict]:
        """Build method.

        Returns:
            list[FlextTypes.Core.Dict]: Copy of the test cases.

        """
        return self._cases.copy()

    def build_pytest_params(self) -> list[tuple[str, str, bool]]:
        """build_pytest_params method.

        Returns:
            list[tuple[str, str, bool]]: Pytest parameters for testing.

        """
        success_params = [
            (str(c.get("email", "")), str(c.get("input", "")), True)
            for c in self._success_cases
        ]
        failure_params = [
            (str(c.get("email", "")), str(c.get("input", "")), False)
            for c in self._failure_cases
        ]
        return success_params + failure_params

    def build_test_ids(self) -> FlextTypes.Core.StringList:
        """build_test_ids method.

        Returns:
            FlextTypes.Core.StringList: List of test IDs.

        """
        return [
            str(c.get("input", ""))
            for c in (*self._success_cases, *self._failure_cases)
        ]


class AssertionBuilder:
    """Builder for complex test assertions."""

    def __init__(self, data: object) -> None:
        """Initialize assertionbuilder:."""
        self.data = data
        self._assertions: list[Callable[[], None]] = []

    def assert_equals(self, expected: object) -> AssertionBuilder:
        """assert_equals method.

        Returns:
            AssertionBuilder: Self for method chaining.

        """

        def assertion() -> None:
            assert self.data == expected

        self._assertions.append(assertion)
        return self

    def assert_contains(self, item: object) -> AssertionBuilder:
        """assert_contains method.

        Returns:
            AssertionBuilder: Self for method chaining.

        """

        def assertion() -> None:
            # Check if item is in the data container
            if isinstance(self.data, (list, tuple, set, str)):
                # Type-safe check for sequence types
                if isinstance(self.data, (list, tuple, set)):
                    assert item in self.data
                elif isinstance(self.data, str):
                    # For strings, check if item is also a string
                    if isinstance(item, str):
                        assert item in self.data
                    else:
                        raise AssertionError(
                            f"Cannot check if {item!r} is in string {self.data!r}",
                        )
            elif isinstance(self.data, dict):
                # For dict, check if item is a key
                assert item in self.data
            else:
                # For other types, check if the item is equal to the data
                assert item == self.data

        self._assertions.append(assertion)
        return self

    def assert_type(self, expected_type: type) -> AssertionBuilder:
        """assert_type method.

        Returns:
            AssertionBuilder: Self for method chaining.

        """

        def assertion() -> None:
            assert isinstance(self.data, expected_type)

        self._assertions.append(assertion)
        return self

    def satisfies(
        self, condition: Callable[[object], bool], description: str = ""
    ) -> AssertionBuilder:
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


class TestAdvancedPatterns:
    """Test class demonstrating advanced testing patterns."""

    def test_given_when_then_builder_pattern(self) -> None:
        """Test Given-When-Then builder pattern."""
        scenario = (
            GivenWhenThenBuilder("user_registration")
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
            FlextTestBuilder()
            .with_id("test-123")
            .with_correlation_id("corr-456")
            .with_user_data("John Doe", "john@example.com")
            .with_timestamp()
            .with_metadata(environment="test", version="1.0")
            .build()
        )

        assert test_data["id"] == "test-123"
        assert test_data["correlation_id"] == "corr-456"
        assert test_data["name"] == "John Doe"
        assert test_data["email"] == "john@example.com"
        assert "created_at" in test_data
        assert "updated_at" in test_data
        assert test_data["environment"] == "test"
        assert test_data["version"] == "1.0"

    def test_parameterized_test_builder_pattern(self) -> None:
        """Test parameterized test builder pattern."""
        builder = (
            ParameterizedTestBuilder("email_validation")
            .add_success_cases(
                [
                    {"email": "valid@example.com", "input": "valid@example.com"},
                    {
                        "email": "user.name@domain.co.uk",
                        "input": "user.name@domain.co.uk",
                    },
                ],
            )
            .add_failure_cases(
                [
                    {"email": "invalid-email", "input": "invalid-email"},
                    {"email": "@domain.com", "input": "@domain.com"},
                ],
            )
        )

        params = builder.build_pytest_params()
        assert len(params) == 4
        assert params[0][2] is True  # success case
        assert params[2][2] is False  # failure case

        test_ids = builder.build_test_ids()
        assert len(test_ids) == 4
        assert "valid@example.com" in test_ids
        assert "invalid-email" in test_ids

    def test_assertion_builder_pattern(self) -> None:
        """Test assertion builder pattern."""
        test_data = {"name": "John", "age": 30, "active": True}

        assertion_builder = (
            AssertionBuilder(test_data)
            .assert_type(dict)
            .assert_contains("name")
            .assert_equals({"name": "John", "age": 30, "active": True})
        )

        # All assertions should pass
        assertion_builder.execute_all()

    @mark_test_pattern("mock_scenario")
    def test_mock_scenario_pattern(self) -> None:
        """Test mock scenario pattern."""
        scenario_data = {
            "given": {"user": "authenticated"},
            "when": {"action": "request_data"},
            "then": {"result": "success"},
            "tags": ["api", "integration"],
            "priority": "medium",
        }

        scenario = MockScenario(
            "api_request",
            cast("FlextTypes.Core.Dict", scenario_data),
        )

        assert scenario.name == "api_request"
        assert scenario.given["user"] == "authenticated"
        assert scenario.when["action"] == "request_data"
        assert scenario.then["result"] == "success"
        assert "api" in scenario.tags
        assert scenario.priority == "medium"

    def test_advanced_mock_patterns(self) -> None:
        """Test advanced mock patterns."""
        # Create a mock with side effects
        mock_service = Mock()
        mock_service.process.side_effect = [
            {"status": "processing"},
            {"status": "completed"},
            Exception("Service unavailable"),
        ]

        # Test successful calls
        result1 = mock_service.process("data1")
        result2 = mock_service.process("data2")

        assert result1["status"] == "processing"
        assert result2["status"] == "completed"
        assert mock_service.process.call_count == 2

        # Test exception handling
        with pytest.raises(Exception, match="Service unavailable"):
            mock_service.process("data3")

    def test_complex_test_data_generation(self) -> None:
        """Test complex test data generation."""
        # Generate multiple test scenarios
        scenarios: list[MockScenario] = []
        for i in range(3):
            scenario = (
                GivenWhenThenBuilder(f"scenario_{i}")
                .given(f"setup_{i}", value=i)
                .when(f"action_{i}", operation=f"op_{i}")
                .then(f"result_{i}", outcome=f"outcome_{i}")
                .build()
            )
            scenarios.append(scenario)

        assert len(scenarios) == 3
        assert scenarios[0].name == "scenario_0"
        assert scenarios[1].given["value"] == 1
        assert scenarios[2].when["operation"] == "op_2"

    def test_nested_builder_patterns(self) -> None:
        """Test nested builder patterns."""
        # Create a complex nested structure
        main_data = (
            FlextTestBuilder()
            .with_id("main-123")
            .with_metadata(
                nested_data=FlextTestBuilder()
                .with_id("nested-456")
                .with_user_data("Jane", "jane@example.com")
                .build(),
            )
            .build()
        )

        assert main_data["id"] == "main-123"
        assert isinstance(main_data["nested_data"], dict)
        assert main_data["nested_data"]["id"] == "nested-456"
        assert main_data["nested_data"]["name"] == "Jane"

    def test_fluent_interface_pattern(self) -> None:
        """Test fluent interface pattern."""
        # Demonstrate fluent interface with method chaining
        result = (
            AssertionBuilder([1, 2, 3, 4, 5])
            .assert_type(list)
            .assert_contains(3)
            .satisfies(
                lambda x: len(x) == 5
                if hasattr(x, "__len__") and isinstance(x, (list, tuple, str, dict))
                else False,
                "should have 5 elements",
            )
        )

        # Execute the assertions
        result.execute_all()

        # Verify method chaining returns self
        assert isinstance(result, AssertionBuilder)

    def test_test_pattern_marking(self) -> None:
        """Test test pattern marking functionality."""
        # This test should have the pattern attribute set
        test_func = self.test_mock_scenario_pattern
        assert hasattr(test_func, "_test_pattern")
        # Use getattr to safely access the attribute
        pattern_value = getattr(test_func, "_test_pattern", None)
        assert pattern_value == "mock_scenario"
