"""Advanced test patterns and builder utilities.

Provides sophisticated test patterns including the Builder pattern, Arrange-Act-Assert,
Given-When-Then, test data builders, and comprehensive test scenario management.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeVar

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class TestScenario:
    """Represents a complete test scenario with metadata."""

    name: str
    description: str
    given: dict[str, Any] = field(default_factory=dict)
    when: dict[str, Any] = field(default_factory=dict)
    then: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    priority: str = "medium"  # low, medium, high, critical
    estimated_duration_ms: int = 1000
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add_tag(self, tag: str) -> TestScenario:
        """Add a tag to the scenario."""
        if tag not in self.tags:
            self.tags.append(tag)
        return self

    def set_priority(self, priority: str) -> TestScenario:
        """Set scenario priority."""
        self.priority = priority
        return self

    def set_duration(self, ms: int) -> TestScenario:
        """Set estimated duration."""
        self.estimated_duration_ms = ms
        return self


class TestDataBuilder[T]:
    """Generic builder pattern for test data construction."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._validators: list[Callable[[dict[str, Any]], bool]] = []
        self._transformers: list[Callable[[dict[str, Any]], dict[str, Any]]] = []

    def with_field(self, name: str, value: Any) -> TestDataBuilder[T]:
        """Add a field to the builder."""
        self._data[name] = value
        return self

    def with_validator(self, validator: Callable[[dict[str, Any]], bool]) -> TestDataBuilder[T]:
        """Add a validator function."""
        self._validators.append(validator)
        return self

    def with_transformer(
        self, transformer: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> TestDataBuilder[T]:
        """Add a transformer function."""
        self._transformers.append(transformer)
        return self

    def build(self) -> dict[str, Any]:
        """Build the final data object."""
        # Apply transformers
        result = self._data.copy()
        for transformer in self._transformers:
            result = transformer(result)

        # Run validators
        for validator in self._validators:
            if not validator(result):
                raise ValueError(f"Validation failed for data: {result}")

        return result

    def build_many(self, count: int) -> list[dict[str, Any]]:
        """Build multiple instances."""
        return [self.build() for _ in range(count)]


class FlextTestBuilder(TestDataBuilder[dict[str, Any]]):
    """Specialized builder for Flext test data."""

    def with_id(self, id_value: str | None = None) -> FlextTestBuilder:
        """Add an ID field."""
        from uuid import uuid4
        actual_id = id_value or f"test_{uuid4().hex[:8]}"
        return self.with_field("id", actual_id)

    def with_correlation_id(self, corr_id: str | None = None) -> FlextTestBuilder:
        """Add a correlation ID."""
        from uuid import uuid4
        actual_corr_id = corr_id or f"corr_{uuid4().hex[:8]}"
        return self.with_field("correlation_id", actual_corr_id)

    def with_timestamp(self, timestamp: datetime | None = None) -> FlextTestBuilder:
        """Add timestamp fields."""
        actual_timestamp = timestamp or datetime.now(UTC)
        return (
            self.with_field("created_at", actual_timestamp)
            .with_field("updated_at", actual_timestamp)
        )

    def with_user_data(self, name: str | None = None, email: str | None = None) -> FlextTestBuilder:
        """Add user-related data."""
        actual_name = name or "Test User"
        actual_email = email or "test.user@example.com"
        return (
            self.with_field("name", actual_name)
            .with_field("email", actual_email)
        )

    def with_validation_rules(self) -> FlextTestBuilder:
        """Add common validation rules."""
        def validate_required_fields(data: dict[str, Any]) -> bool:
            required = ["id"] if "id" in self._data else []
            return all(field in data for field in required)

        def validate_email_format(data: dict[str, Any]) -> bool:
            if "email" in data:
                return "@" in str(data["email"])
            return True

        return (
            self.with_validator(validate_required_fields)
            .with_validator(validate_email_format)
        )


class GivenWhenThenBuilder:
    """Builder for Given-When-Then style test scenarios."""

    def __init__(self, name: str) -> None:
        self.scenario = TestScenario(name=name, description="")
        self._given_steps: list[str] = []
        self._when_steps: list[str] = []
        self._then_steps: list[str] = []

    def given(self, description: str, **context: Any) -> GivenWhenThenBuilder:
        """Add a Given step."""
        self._given_steps.append(description)
        self.scenario.given.update(context)
        return self

    def when(self, description: str, **context: Any) -> GivenWhenThenBuilder:
        """Add a When step."""
        self._when_steps.append(description)
        self.scenario.when.update(context)
        return self

    def then(self, description: str, **context: Any) -> GivenWhenThenBuilder:
        """Add a Then step."""
        self._then_steps.append(description)
        self.scenario.then.update(context)
        return self

    def with_tag(self, tag: str) -> GivenWhenThenBuilder:
        """Add a tag."""
        self.scenario.add_tag(tag)
        return self

    def with_priority(self, priority: str) -> GivenWhenThenBuilder:
        """Set priority."""
        self.scenario.set_priority(priority)
        return self

    def build(self) -> TestScenario:
        """Build the complete scenario."""
        # Create description from steps
        description_parts = []
        if self._given_steps:
            description_parts.append(f"Given: {'; '.join(self._given_steps)}")
        if self._when_steps:
            description_parts.append(f"When: {'; '.join(self._when_steps)}")
        if self._then_steps:
            description_parts.append(f"Then: {'; '.join(self._then_steps)}")

        self.scenario.description = " | ".join(description_parts)
        return self.scenario


class TestCaseFactory:
    """Factory for creating various types of test cases."""

    @staticmethod
    def create_success_case(
        name: str,
        input_data: dict[str, Any],
        expected_output: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a success test case."""
        return {
            "name": name,
            "type": "success",
            "input": input_data,
            "expected": expected_output,
            "should_succeed": True,
            "tags": ["happy_path", "success"]
        }

    @staticmethod
    def create_failure_case(
        name: str,
        input_data: dict[str, Any],
        expected_error: str
    ) -> dict[str, Any]:
        """Create a failure test case."""
        return {
            "name": name,
            "type": "failure",
            "input": input_data,
            "expected_error": expected_error,
            "should_succeed": False,
            "tags": ["error_path", "failure"]
        }

    @staticmethod
    def create_edge_case(
        name: str,
        input_data: dict[str, Any],
        expected_behavior: str
    ) -> dict[str, Any]:
        """Create an edge case test."""
        return {
            "name": name,
            "type": "edge_case",
            "input": input_data,
            "expected_behavior": expected_behavior,
            "tags": ["edge_case", "boundary"]
        }

    @staticmethod
    def create_performance_case(
        name: str,
        input_data: dict[str, Any],
        max_duration_ms: int
    ) -> dict[str, Any]:
        """Create a performance test case."""
        return {
            "name": name,
            "type": "performance",
            "input": input_data,
            "max_duration_ms": max_duration_ms,
            "tags": ["performance", "benchmark"]
        }


class ParameterizedTestBuilder:
    """Builder for parametrized tests with automatic case generation."""

    def __init__(self, test_name: str) -> None:
        self.test_name = test_name
        self.parameters: list[dict[str, Any]] = []
        self.parameter_names: list[str] = []

    def add_case(self, **kwargs: Any) -> ParameterizedTestBuilder:
        """Add a single test case."""
        if not self.parameter_names:
            self.parameter_names = list(kwargs.keys())
        # Ensure all cases have the same parameters
        elif set(kwargs.keys()) != set(self.parameter_names):
            raise ValueError(f"Parameter mismatch. Expected: {self.parameter_names}")

        self.parameters.append(kwargs)
        return self

    def add_success_cases(self, cases: list[dict[str, Any]]) -> ParameterizedTestBuilder:
        """Add multiple success cases."""
        for case in cases:
            case["expected_success"] = True
            self.add_case(**case)
        return self

    def add_failure_cases(self, cases: list[dict[str, Any]]) -> ParameterizedTestBuilder:
        """Add multiple failure cases."""
        for case in cases:
            case["expected_success"] = False
            self.add_case(**case)
        return self

    def add_edge_cases(self, cases: list[dict[str, Any]]) -> ParameterizedTestBuilder:
        """Add edge cases with special handling."""
        for case in cases:
            case["is_edge_case"] = True
            self.add_case(**case)
        return self

    def build_pytest_params(self) -> list[tuple[Any, ...]]:
        """Build parameters for pytest.mark.parametrize."""
        return [tuple(case[name] for name in self.parameter_names) for case in self.parameters]

    def build_test_ids(self) -> list[str]:
        """Build test IDs for pytest."""
        return [
            f"{self.test_name}_{i}_{self._generate_case_id(case)}"
            for i, case in enumerate(self.parameters)
        ]

    def _generate_case_id(self, case: dict[str, Any]) -> str:
        """Generate a readable ID for a test case."""
        # Try to create meaningful ID from case data
        if "name" in case:
            return str(case["name"]).replace(" ", "_").lower()
        if "input" in case and isinstance(case["input"], dict):
            key_vals = []
            for k, v in list(case["input"].items())[:2]:  # First 2 items
                key_vals.append(f"{k}_{str(v)[:10]}")
            return "_".join(key_vals).replace(" ", "_").lower()
        return "case"


class TestFixtureBuilder:
    """Builder for creating test fixtures and setup data."""

    def __init__(self) -> None:
        self.fixtures: dict[str, Any] = {}
        self.setup_functions: list[Callable[[], None]] = []
        self.teardown_functions: list[Callable[[], None]] = []

    def add_fixture(self, name: str, value: Any) -> TestFixtureBuilder:
        """Add a fixture value."""
        self.fixtures[name] = value
        return self

    def add_setup(self, func: Callable[[], None]) -> TestFixtureBuilder:
        """Add a setup function."""
        self.setup_functions.append(func)
        return self

    def add_teardown(self, func: Callable[[], None]) -> TestFixtureBuilder:
        """Add a teardown function."""
        self.teardown_functions.append(func)
        return self

    @contextmanager
    def setup_context(self):
        """Context manager for test setup and teardown."""
        # Run setup
        for setup_func in self.setup_functions:
            setup_func()

        try:
            yield self.fixtures
        finally:
            # Run teardown
            for teardown_func in self.teardown_functions:
                try:
                    teardown_func()
                except Exception:
                    pass  # Don't let teardown errors mask test failures


class TestAssertionBuilder:
    """Builder for complex assertion patterns."""

    def __init__(self, actual_value: Any) -> None:
        self.actual = actual_value
        self.assertions: list[Callable[[Any], bool]] = []
        self.descriptions: list[str] = []

    def is_equal_to(self, expected: Any) -> TestAssertionBuilder:
        """Assert equality."""
        self.assertions.append(lambda x: x == expected)
        self.descriptions.append(f"should equal {expected}")
        return self

    def is_not_none(self) -> TestAssertionBuilder:
        """Assert not None."""
        self.assertions.append(lambda x: x is not None)
        self.descriptions.append("should not be None")
        return self

    def has_length(self, length: int) -> TestAssertionBuilder:
        """Assert length."""
        self.assertions.append(lambda x: len(x) == length)
        self.descriptions.append(f"should have length {length}")
        return self

    def contains(self, item: Any) -> TestAssertionBuilder:
        """Assert contains item."""
        self.assertions.append(lambda x: item in x)
        self.descriptions.append(f"should contain {item}")
        return self

    def matches_pattern(self, pattern: str) -> TestAssertionBuilder:
        """Assert regex pattern match."""
        import re
        self.assertions.append(lambda x: bool(re.match(pattern, str(x))))
        self.descriptions.append(f"should match pattern {pattern}")
        return self

    def satisfies(self, predicate: Callable[[Any], bool], description: str) -> TestAssertionBuilder:
        """Assert custom predicate."""
        self.assertions.append(predicate)
        self.descriptions.append(description)
        return self

    def assert_all(self) -> None:
        """Execute all assertions."""
        for i, (assertion, description) in enumerate(zip(self.assertions, self.descriptions, strict=False)):
            if not assertion(self.actual):
                raise AssertionError(
                    f"Assertion {i + 1} failed: {description}. "
                    f"Actual value: {self.actual}"
                )


def test_pattern(pattern_name: str):
    """Decorator to mark tests with specific patterns."""
    def decorator(func: Callable) -> Callable:
        func._test_pattern = pattern_name
        return func
    return decorator


def arrange_act_assert(arrange_func: Callable, act_func: Callable, assert_func: Callable):
    """Decorator that enforces the Arrange-Act-Assert pattern."""
    def decorator(test_func: Callable) -> Callable:
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            # Arrange
            arranged_data = arrange_func(*args, **kwargs)

            # Act
            result = act_func(arranged_data)

            # Assert
            assert_func(result, arranged_data)

            return result

        return wrapper
    return decorator


class TestSuiteBuilder:
    """Builder for organizing multiple related tests."""

    def __init__(self, suite_name: str) -> None:
        self.suite_name = suite_name
        self.test_scenarios: list[TestScenario] = []
        self.setup_data: dict[str, Any] = {}
        self.tags: list[str] = []

    def add_scenario(self, scenario: TestScenario) -> TestSuiteBuilder:
        """Add a test scenario."""
        self.test_scenarios.append(scenario)
        return self

    def add_scenarios(self, scenarios: list[TestScenario]) -> TestSuiteBuilder:
        """Add multiple scenarios."""
        self.test_scenarios.extend(scenarios)
        return self

    def with_setup_data(self, **data: Any) -> TestSuiteBuilder:
        """Add setup data for the suite."""
        self.setup_data.update(data)
        return self

    def with_tag(self, tag: str) -> TestSuiteBuilder:
        """Add a tag to all scenarios in the suite."""
        self.tags.append(tag)
        for scenario in self.test_scenarios:
            scenario.add_tag(tag)
        return self

    def build(self) -> dict[str, Any]:
        """Build the complete test suite."""
        return {
            "suite_name": self.suite_name,
            "scenarios": self.test_scenarios,
            "setup_data": self.setup_data,
            "tags": self.tags,
            "scenario_count": len(self.test_scenarios),
            "estimated_duration_ms": sum(s.estimated_duration_ms for s in self.test_scenarios)
        }
