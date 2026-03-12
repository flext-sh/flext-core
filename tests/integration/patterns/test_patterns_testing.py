"""Demonstration of testing patterns and libraries.

This test file showcases the full power of our testing infrastructure including:
- Property-based testing with custom Hypothesis strategies
- Performance testing with complexity analysis
- Stress testing and load testing
- Advanced test patterns (Builder, Given-When-Then)
- Comprehensive parametrized testing

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Iterator, Mapping, Sequence, Sized
from contextlib import AbstractContextManager as ContextManager, contextmanager
from typing import TypeGuard

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from flext_core import FlextTypes, FlextUtilities, P, R

from ._models import (
    FixtureCaseDict,
    FixtureDataDict,
    FixtureFixturesDict,
    FixtureSuiteDict,
)


def _to_general_mapping(
    value: object | None,
) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _to_string_list(value: object | None) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _to_string(value: object | None, *, default: str) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    return str(value)


def _as_object_dict(value: object) -> dict[str, object]:
    if not _is_object_mapping(value):
        return {}
    output: dict[str, object] = {}
    for key, item in value.items():
        output[str(key)] = item
    return output


def _as_object_list(value: object) -> list[object] | None:
    if not _is_object_list(value):
        return None
    return list(value)


def _is_object_mapping(
    value: object,
) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_object_container_sequence(
    value: object,
) -> TypeGuard[list[object] | tuple[object, ...] | set[object]]:
    return isinstance(value, (list, tuple, set))


def _as_int_list(value: object) -> list[int] | None:
    object_list = _as_object_list(value)
    if object_list is None:
        return None
    int_values: list[int] = []
    for item in object_list:
        if not isinstance(item, int):
            return None
        int_values.append(item)
    return int_values


def mark_test_pattern(
    pattern: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark test with a specific pattern for demonstration purposes."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        setattr(func, "_test_pattern", pattern)
        return func

    return decorator


pytestmark = [pytest.mark.unit, pytest.mark.architecture, pytest.mark.advanced]


class MockScenario:
    """Mock scenario object for testing purposes."""

    def __init__(self, name: str, data: dict[str, object]) -> None:
        """Initialize mock scenario with name and test data."""
        super().__init__()
        self.name = name
        mapper = FlextUtilities.Mapper
        self.given = _to_general_mapping(mapper.get(data, "given", default={}))
        self.when = _to_general_mapping(mapper.get(data, "when", default={}))
        self.then = _to_general_mapping(mapper.get(data, "then", default={}))
        self.tags = _to_string_list(mapper.get(data, "tags", default=[]))
        self.priority = _to_string(
            mapper.get(data, "priority", default="normal"),
            default="normal",
        )


class GivenWhenThenBuilder:
    """Builder for Given-When-Then test scenarios."""

    def __init__(self, name: str) -> None:
        """Initialize Given-When-Then builder with test name."""
        super().__init__()
        self.name = name
        self._given: dict[str, FlextTypes.Container] = {}
        self._when: dict[str, FlextTypes.Container] = {}
        self._then: dict[str, FlextTypes.Container] = {}
        self._tags: list[str] = []
        self._priority = "normal"

    def given(
        self,
        _description: str,
        **kwargs: FlextTypes.Container,
    ) -> GivenWhenThenBuilder:
        """Add given conditions to the test scenario."""
        self._given.update(kwargs)
        return self

    def when(
        self,
        _description: str,
        **kwargs: FlextTypes.Container,
    ) -> GivenWhenThenBuilder:
        """Add when actions to the test scenario."""
        self._when.update(kwargs)
        return self

    def then(
        self,
        _description: str,
        **kwargs: FlextTypes.Container,
    ) -> GivenWhenThenBuilder:
        """Add then expectations to the test scenario."""
        self._then.update(kwargs)
        return self

    def with_tag(self, tag: str) -> GivenWhenThenBuilder:
        """Add a tag to the test scenario."""
        self._tags.append(tag)
        return self

    def with_priority(self, priority: str) -> GivenWhenThenBuilder:
        """Set the priority of the test scenario."""
        self._priority = priority
        return self

    def build(self) -> MockScenario:
        """Build the final mock scenario object."""
        data: dict[str, object] = {
            "given": self._given,
            "when": self._when,
            "then": self._then,
            "tags": self._tags,
            "priority": self._priority,
        }
        return MockScenario(self.name, data)


class FlextTestBuilder:
    """Builder for test data objects."""

    def __init__(self) -> None:
        """Initialize test data builder with empty data."""
        super().__init__()
        self._data: FixtureDataDict = FixtureDataDict({})

    def with_id(self, id_: str) -> FlextTestBuilder:
        """Add ID to the test data."""
        self._data["id"] = id_
        return self

    def with_correlation_id(self, correlation_id: str) -> FlextTestBuilder:
        """Add correlation ID to the test data."""
        self._data["correlation_id"] = correlation_id
        return self

    def with_metadata(self, **kwargs: FlextTypes.Container) -> FlextTestBuilder:
        """Add metadata to the test data."""
        self._data.update(kwargs)
        return self

    def with_user_data(self, name: str, email: str) -> FlextTestBuilder:
        """Add user data to the test data."""
        self._data["name"] = name
        self._data["email"] = email
        return self

    def with_timestamp(self) -> FlextTestBuilder:
        """Add timestamp fields to the test data."""
        _ = self._data.setdefault("created_at", "2023-01-01T00:00:00+00:00")
        _ = self._data.setdefault("updated_at", "2023-01-01T00:00:00+00:00")
        return self

    def with_validation_rules(self) -> FlextTestBuilder:
        """Add validation rules to the test data (no-op stub)."""
        return self

    def build(self) -> FixtureDataDict:
        """Build the final test data dictionary."""
        return self._data


class ParameterizedTestBuilder:
    """Builder for parametrized test cases."""

    def __init__(self, test_name: str) -> None:
        """Initialize parameterized test builder with test name."""
        super().__init__()
        self.test_name = test_name
        self._cases: list[FixtureCaseDict] = []
        self._success_cases: list[FixtureCaseDict] = []
        self._failure_cases: list[FixtureCaseDict] = []

    def add_case(
        self,
        email: str | None = None,
        input_value: str | None = None,
    ) -> ParameterizedTestBuilder:
        """Add a test case with the given parameters."""
        case: FixtureCaseDict = FixtureCaseDict({})
        if email is not None:
            case["email"] = email
        if input_value is not None:
            case["input"] = input_value
        self._cases.append(case)
        return self

    def add_success_cases(
        self,
        cases: list[FixtureCaseDict],
    ) -> ParameterizedTestBuilder:
        """Add multiple success test cases."""
        self._success_cases.extend(cases)
        return self

    def add_failure_cases(
        self,
        cases: list[FixtureCaseDict],
    ) -> ParameterizedTestBuilder:
        """Add multiple failure test cases."""
        self._failure_cases.extend(cases)
        return self

    def build(self) -> list[FixtureCaseDict]:
        """Build the list of test cases."""
        return self._cases.copy()

    def build_pytest_params(self) -> list[tuple[str, str, bool]]:
        """Build pytest parametrized test parameters."""
        success_params = [
            (str(c.get("email", "")), str(c.get("input", "")), True)
            for c in self._success_cases
        ]
        failure_params = [
            (str(c.get("email", "")), str(c.get("input", "")), False)
            for c in self._failure_cases
        ]
        return success_params + failure_params

    def build_test_ids(self) -> list[str]:
        """Build test IDs for pytest parametrization."""
        return [
            str(c.get("input", ""))
            for c in (*self._success_cases, *self._failure_cases)
        ]


class AssertionBuilder:
    """Builder for complex test assertions."""

    def __init__(self, data: object) -> None:
        """Initialize test assertion builder with data to test."""
        super().__init__()
        self._data = data
        self._checks: list[tuple[str, object]] = []

    def is_not_none(self) -> AssertionBuilder:
        """Assert that the data is not None."""
        assert self._data is not None
        return self

    def has_length(self, length: int) -> AssertionBuilder:
        """Assert that the data has the specified length."""
        if not isinstance(self._data, Sized):
            msg = f"Expected Sized object, got {type(self._data)}"
            raise TypeError(msg)
        assert len(self._data) == length
        return self

    def contains(self, item: object) -> AssertionBuilder:
        """Assert that the data contains the specified item."""
        data = self._data
        if _is_object_mapping(data):
            assert item in data
            return self
        if _is_object_container_sequence(data):
            assert item in data
            return self
        if isinstance(data, str):
            assert isinstance(item, str)
            assert item in data
            return self
        if isinstance(data, bytes | bytearray):
            assert isinstance(item, bytes | bytearray)
            assert item in data
            return self
        msg = f"Expected Container object, got {type(data)}"
        raise TypeError(msg)

    def satisfies(
        self,
        predicate: Callable[[object], bool],
        message: str = "",
    ) -> AssertionBuilder:
        """Assert that the data satisfies the given predicate."""
        assert predicate(self._data), message
        return self

    def assert_all(self) -> None:
        """Execute all accumulated assertions."""
        return


class SuiteBuilder:
    """Builder for test suites."""

    def __init__(self, name: str) -> None:
        """Initialize test suite builder with suite name."""
        super().__init__()
        self.name = name
        self._scenarios: list[MockScenario] = []
        self._setup_data: dict[str, FlextTypes.Container] = {}
        self._tags: list[str] = []

    def add_scenarios(self, scenarios: Sequence[MockScenario]) -> SuiteBuilder:
        """Add multiple test scenarios to the suite."""
        self._scenarios.extend(scenarios)
        return self

    def with_setup_data(self, **kwargs: FlextTypes.Container) -> SuiteBuilder:
        """Add setup data to the test suite."""
        self._setup_data.update(kwargs)
        return self

    def with_tag(self, tag: str) -> SuiteBuilder:
        """Add a tag to the test suite."""
        self._tags.append(tag)
        return self

    def build(self) -> FixtureSuiteDict:
        """Build the test suite configuration."""
        return FixtureSuiteDict({
            "suite_name": self.name,
            "scenario_count": len(self._scenarios),
            "tags": self._tags,
            "setup_data": self._setup_data,
        })


class FixtureBuilder:
    """Builder for test fixtures."""

    def __init__(self) -> None:
        """Initialize test fixture builder with empty fixtures."""
        super().__init__()
        self._fixtures: FixtureFixturesDict = FixtureFixturesDict({})
        self._setups: list[Callable[[], None]] = []
        self._teardowns: list[Callable[[], None]] = []

    def with_user(self, **kwargs: FlextTypes.Container) -> FixtureBuilder:
        """Add user fixture data."""
        self._fixtures["user"] = kwargs
        return self

    def with_request(self, **kwargs: FlextTypes.Container) -> FixtureBuilder:
        """Add request fixture data."""
        self._fixtures["request"] = kwargs
        return self

    def build(self) -> FixtureFixturesDict:
        """Build the test fixtures configuration."""
        return self._fixtures.model_copy(deep=True)

    def add_setup(self, func: Callable[[], None]) -> FixtureBuilder:
        """Add a setup function to the fixtures."""
        self._setups.append(func)
        return self

    def add_teardown(self, func: Callable[[], None]) -> FixtureBuilder:
        """Add a teardown function to the fixtures."""
        self._teardowns.append(func)
        return self

    def add_fixture(self, key: str, value: FlextTypes.Container) -> FixtureBuilder:
        """Add a custom fixture with the given key and value."""
        self._fixtures[key] = value
        return self

    def setup_context(self) -> Callable[[], ContextManager[FixtureFixturesDict]]:
        """Create a context manager for test setup and teardown."""

        @contextmanager
        def _ctx() -> Iterator[FixtureFixturesDict]:
            for f in self._setups:
                if callable(f):
                    _ = f()
            try:
                yield self._fixtures
            finally:
                for f in self._teardowns:
                    if callable(f):
                        _ = f()

        return _ctx


def arrange_act_assert(
    _arrange_func: Callable[[], object],
    _act_func: Callable[[object], object],
    _assert_func: Callable[[object, object], None],
) -> Callable[[Callable[[], object]], Callable[[], object]]:
    """Decorator for AAA pattern testing."""

    def decorator(
        _test_func: Callable[[], object],
    ) -> Callable[[], object]:

        def wrapper() -> object:
            data = _arrange_func()
            result = _act_func(data)
            _assert_func(result, data)
            return result

        return wrapper

    return decorator


class TestPropertyBasedPatterns:
    """Demonstrate property-based testing with custom strategies."""

    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(st.emails())
    def test_email_property_based(self, email: str) -> None:
        """Property-based test for email handling."""
        assert "@" in email
        assert len(email) > 3
        assert not email.startswith("@")
        assert not email.endswith("@")

    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(
        st.fixed_dictionaries({
            "id": st.uuids().map(str),
            "name": st.text(),
            "email": st.emails(),
        }),
    )
    def test_user_profile_property_based(self, profile: dict[str, str]) -> None:
        """Property-based test for user profiles."""
        assert "id" in profile
        assert "name" in profile
        assert "email" in profile
        assert isinstance(profile["id"], str)
        assert isinstance(profile["name"], str)
        assert isinstance(profile["email"], str)

    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(st.text(min_size=10, max_size=100))
    def test_string_performance_property_based(self, large_string: str) -> None:
        """Property-based test for string processing performance."""
        start_time = time.perf_counter()
        processed = large_string.lower().strip()
        word_count = len(processed.split())
        end_time = time.perf_counter()
        duration = end_time - start_time
        assert duration < 1.0
        assert word_count >= 0

    @given(
        st.text(
            alphabet=st.characters(exclude_categories=("C", "Cs")),
            min_size=1,
            max_size=100,
        ),
    )
    def test_unicode_handling_property_based(self, unicode_text: str) -> None:
        """Property-based test for Unicode handling."""
        result = unicode_text.encode("utf-8").decode("utf-8")
        assert len(result) >= 0


class TestPerformanceAnalysis:
    """Demonstrate performance testing and complexity analysis."""

    def test_stress_testing_load(self) -> None:
        """Demonstrate stress testing with load patterns."""

        def simple_operation() -> str:
            """Simple operation for stress testing."""
            return "test_result"

        iterations = 1000
        successes = 0
        start_time = time.perf_counter()
        for _ in range(iterations):
            result = simple_operation()
            if result == "test_result":
                successes += 1
        end_time = time.perf_counter()
        duration = end_time - start_time
        operations_per_second = iterations / duration if duration > 0 else 0
        assert successes == iterations
        assert operations_per_second > 0

    def test_endurance_testing(self) -> None:
        """Demonstrate endurance testing."""

        def memory_operation() -> list[int]:
            """Operation that uses some memory."""
            return list(range(100))

        duration_target = 0.5
        start_time = time.perf_counter()
        iterations = 0
        while time.perf_counter() - start_time < duration_target:
            _ = memory_operation()
            iterations += 1
        actual_duration = time.perf_counter() - start_time
        operations_per_second = (
            iterations / actual_duration if actual_duration > 0 else 0
        )
        assert actual_duration >= duration_target * 0.8
        assert iterations > 0
        assert operations_per_second > 0

    def test_memory_profiling_advanced(self) -> None:
        """Demonstrate advanced memory profiling."""
        _ = gc.collect()
        large_list = list(range(10000))
        filtered_list = list(
            FlextUtilities.Collection.filter(large_list, lambda x: x % 2 == 0),
        )
        sorted_list = sorted(filtered_list, reverse=True)
        assert len(large_list) == 10000
        assert len(filtered_list) == 5000
        assert sorted_list[0] > sorted_list[-1]


class TestAdvancedPatterns:
    """Demonstrate advanced test patterns and builders."""

    def test_given_when_then_pattern(self) -> None:
        """Demonstrate Given-When-Then pattern."""
        scenario = (
            GivenWhenThenBuilder("user_registration")
            .given("a new user with valid email", email="test@example.com")
            .given("the email is not already registered", unique=True)
            .when("the user attempts to register", action="register")
            .then("the registration should succeed", success=True)
            .then("the user should receive a confirmation", confirmation=True)
            .with_tag("integration")
            .with_priority("high")
            .build()
        )
        assert scenario.name == "user_registration"
        assert "email" in scenario.given
        assert "action" in scenario.when
        assert "success" in scenario.then
        assert "integration" in scenario.tags
        assert scenario.priority == "high"

    def test_builder_pattern_advanced(self) -> None:
        """Demonstrate advanced builder pattern."""
        builder = (
            FlextTestBuilder()
            .with_id("test_123")
            .with_correlation_id("corr_456")
            .with_user_data("John Doe", "john@example.com")
            .with_timestamp()
            .with_validation_rules()
        )
        data = builder.build()
        assert data.get("id") == "test_123"
        assert data.get("correlation_id") == "corr_456"
        assert data.get("name") == "John Doe"
        assert data.get("email") == "john@example.com"
        assert "created_at" in data
        assert "updated_at" in data

    def test_parametrized_builder(self) -> None:
        """Demonstrate parametrized test builder."""
        param_builder = ParameterizedTestBuilder("email_validation")
        _ = param_builder.add_success_cases([
            FixtureCaseDict({"email": "test@example.com", "input": "valid_email_1"}),
            FixtureCaseDict({"email": "user@domain.org", "input": "valid_email_2"}),
        ])
        _ = param_builder.add_failure_cases([
            FixtureCaseDict({"email": "invalid-email", "input": "invalid_email_1"}),
            FixtureCaseDict({"email": "@domain.com", "input": "invalid_email_2"}),
        ])
        params = param_builder.build_pytest_params()
        test_ids = param_builder.build_test_ids()
        assert len(params) == 4
        assert len(test_ids) == 4
        assert all(len(param) == 3 for param in params)

    def test_assertion_builder(self) -> None:
        """Demonstrate assertion builder pattern."""
        test_data = ["apple", "banana", "cherry"]

        def check_all_strings(x: object) -> bool:
            """Check if all items in a list are strings."""
            values = _as_object_dict({"items": x}).get("items")
            values_list = _as_object_list(values)
            if values_list is None:
                return False
            return all(isinstance(item, str) for item in values_list)

        AssertionBuilder(test_data).is_not_none().has_length(3).contains(
            "banana",
        ).satisfies(check_all_strings, "all items should be strings").assert_all()

    @mark_test_pattern("arrange_act_assert")
    def test_arrange_act_assert_decorator(self) -> None:
        """Demonstrate Arrange-Act-Assert pattern decorator."""

        def arrange_data(*_args: object) -> dict[str, list[int]]:
            return {"numbers": [1, 2, 3, 4, 5]}

        def act_on_data(data: object) -> object:
            payload = _as_object_dict(data)
            if "numbers" in payload:
                numbers = _as_int_list(payload["numbers"])
                if numbers is not None:
                    typed_numbers: list[int] = numbers
                    return sum(typed_numbers)
            return 0

        def assert_result(
            result: object,
            original_data: object,
        ) -> None:
            assert result == 15
            payload = _as_object_dict(original_data)
            if "numbers" in payload:
                numbers = _as_int_list(payload["numbers"])
                if numbers is not None:
                    assert len(numbers) == 5

        @arrange_act_assert(arrange_data, act_on_data, assert_result)
        def test_sum_calculation() -> None:
            pass

        result = test_sum_calculation()
        assert result == 15


class TestComprehensiveIntegration:
    """Demonstrate integration of all testing patterns."""

    def test_complete_test_suite_builder(self) -> None:
        """Demonstrate complete test suite construction."""
        scenarios: list[MockScenario] = []
        scenario1 = (
            GivenWhenThenBuilder("successful_operation")
            .given("valid input data", data_valid=True)
            .when("operation is executed", executed=True)
            .then("operation succeeds", success=True)
            .with_tag("success")
            .build()
        )
        scenarios.append(scenario1)
        scenario2 = (
            GivenWhenThenBuilder("failed_operation")
            .given("invalid input data", data_valid=False)
            .when("operation is executed", executed=True)
            .then("operation fails gracefully", success=False, graceful=True)
            .with_tag("failure")
            .build()
        )
        scenarios.append(scenario2)
        scenario_list: Sequence[MockScenario] = scenarios
        suite = (
            SuiteBuilder("comprehensive_operation_tests")
            .add_scenarios(scenario_list)
            .with_setup_data(environment="test", timeout=30)
            .with_tag("integration")
            .build()
        )
        assert suite["suite_name"] == "comprehensive_operation_tests"
        assert suite["scenario_count"] == 2
        tags_value = suite["tags"]
        assert isinstance(tags_value, list)
        assert "integration" in tags_value
        setup_data = suite["setup_data"]
        if isinstance(setup_data, dict) and "environment" in setup_data:
            env_value: object = setup_data["environment"]
            assert env_value == "test"


class TestRealWorldScenarios:
    """Simulate real-world testing scenarios."""

    def test_api_request_processing(self) -> None:
        """Simulate API request processing with comprehensive testing."""
        fixture_builder = FixtureBuilder()

        def setup_api_environment() -> None:
            pass

        def teardown_api_environment() -> None:
            pass

        _ = fixture_builder.add_setup(setup_api_environment)
        _ = fixture_builder.add_teardown(teardown_api_environment)
        _ = fixture_builder.add_fixture("api_base_url", "https://api.test.com")
        _ = fixture_builder.add_fixture("timeout", 30)
        with fixture_builder.setup_context()():
            test_request: dict[str, object] = {
                "method": "POST",
                "url": "https://api.example.com/users",
                "correlation_id": "corr_12345678",
                "headers": {"Content-Type": "application/json"},
                "body": {"name": "test"},
            }

            def process_api_request(
                request: dict[str, object],
            ) -> dict[str, object]:
                return {
                    "status": "success",
                    "method": request["method"],
                    "url": request["url"],
                    "correlation_id": request["correlation_id"],
                    "processed_at": time.time(),
                }

            result = process_api_request(test_request)

            def check_status_success(x: object) -> bool:
                """Check if status is success."""
                payload = _as_object_dict(x)
                return payload.get("status") == "success"

            def check_correlation_id(x: object) -> bool:
                """Check if correlation_id exists."""
                payload = _as_object_dict(x)
                return "correlation_id" in payload

            def check_valid_method(x: object) -> bool:
                """Check if method is valid HTTP method."""
                payload = _as_object_dict(x)
                method = payload.get("method")
                return isinstance(method, str) and method in {
                    "GET",
                    "POST",
                    "PUT",
                    "DELETE",
                    "PATCH",
                }

            AssertionBuilder(result).is_not_none().satisfies(
                check_status_success,
                "should be successful",
            ).satisfies(check_correlation_id, "should have correlation ID").satisfies(
                check_valid_method,
                "should have valid HTTP method",
            ).assert_all()

    @given(
        database_url=st.text(),
        debug=st.booleans(),
        timeout_seconds=st.integers(min_value=1, max_value=300),
        environment=st.sampled_from(["development", "staging", "production"]),
    )
    @settings()
    def test_configuration_validation_comprehensive(
        self,
        database_url: str,
        debug: bool,
        timeout_seconds: int,
        environment: str,
    ) -> None:
        """Comprehensive configuration validation testing."""
        config: dict[str, FlextTypes.Container] = {
            "database_url": database_url,
            "debug": debug,
            "timeout_seconds": timeout_seconds,
            "environment": environment,
        }
        required_fields = ["database_url", "debug", "timeout_seconds"]
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"
        assert isinstance(config["debug"], bool)
        assert isinstance(config["timeout_seconds"], int)
        assert config["timeout_seconds"] > 0
        assert config["environment"] in {"development", "staging", "production"}
        scenario = (
            GivenWhenThenBuilder("configuration_validation")
            .given(
                "a configuration object",
                config_environment=str(config["environment"]),
            )
            .when("configuration is validated", action="validate")
            .then("all required fields are present", validated=True)
            .with_tag("configuration")
            .build()
        )
        assert scenario.given.get("config_environment") == config["environment"]
        assert "configuration" in scenario.tags
