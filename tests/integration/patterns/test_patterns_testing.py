"""Demonstration of testing patterns and libraries.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import gc
import time
from collections.abc import (
    Callable,
    Generator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Sized,
)
from contextlib import AbstractContextManager as ContextManager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import ParamSpec, TypeIs, TypeVar

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from tests import t, u

P = ParamSpec("P")
R = TypeVar("R")

type FixtureCaseDict = t.ContainerMapping
type FixtureDataDict = t.ContainerMapping
type FixtureFixturesDict = t.ContainerMapping
type FixtureSuiteDict = t.ContainerMapping


pytestmark = [pytest.mark.unit, pytest.mark.architecture, pytest.mark.advanced]


class TestPatternsTesting:
    """Unified advanced testing-pattern module with one top-level test class."""

    class Helpers:
        """Helper methods for testing patterns."""

        @staticmethod
        def to_general_mapping(
            value: Mapping[str, t.Container] | None,
        ) -> Mapping[str, t.Container]:
            if value is None:
                return {}
            return dict(value.items())

        @staticmethod
        def to_string_list(value: t.FlatContainerList | None) -> t.StrSequence:
            if value is None:
                return list[str]()
            return [str(item) for item in value]

        @staticmethod
        def to_string(value: t.Container | None, *, default: str) -> str:
            if isinstance(value, str):
                return value
            if value is None:
                return default
            return str(value)

        @staticmethod
        def as_object_dict(value: t.NormalizedValue) -> t.ContainerMapping:
            if not TestPatternsTesting.Helpers.object_mapping(value):
                return dict[str, t.NormalizedValue]()
            output: t.MutableContainerMapping = {}
            for key, item in value.items():
                output[str(key)] = item
            return output

        @staticmethod
        def as_object_list(
            value: t.NormalizedValue,
        ) -> t.ContainerList | None:
            if not TestPatternsTesting.Helpers.object_list(value):
                return None
            return list(value)

        @staticmethod
        def object_mapping(
            value: t.NormalizedValue,
        ) -> TypeIs[t.ContainerMapping]:
            return isinstance(value, Mapping)

        @staticmethod
        def object_list(
            value: t.NormalizedValue,
        ) -> TypeIs[t.ContainerList]:
            return isinstance(value, list)

        @staticmethod
        def object_container_sequence(
            value: t.NormalizedValue,
        ) -> bool:
            return isinstance(value, (list, tuple, set))

        @staticmethod
        def as_int_list(value: t.NormalizedValue) -> Sequence[int] | None:
            object_list = TestPatternsTesting.Helpers.as_object_list(value)
            if object_list is None:
                return None
            int_values: MutableSequence[int] = []
            for item in object_list:
                if not isinstance(item, int):
                    return None
                int_values.append(item)
            return int_values

        @staticmethod
        def mark_test_pattern(
            pattern: str,
        ) -> Callable[[Callable[P, R]], Callable[P, R]]:
            def decorator(func: Callable[P, R]) -> Callable[P, R]:
                setattr(func, "_test_pattern", pattern)
                return func

            return decorator

        @staticmethod
        def arrange_act_assert(
            _arrange_func: Callable[[], t.NormalizedValue],
            _act_func: Callable[[t.NormalizedValue], t.NormalizedValue],
            _assert_func: Callable[[t.NormalizedValue, t.NormalizedValue], None],
        ) -> Callable[
            [Callable[[], t.NormalizedValue]],
            Callable[[], t.NormalizedValue],
        ]:
            def decorator(
                _test_func: Callable[[], t.NormalizedValue],
            ) -> Callable[[], t.NormalizedValue]:
                def wrapper() -> t.NormalizedValue:
                    data = _arrange_func()
                    result = _act_func(data)
                    _assert_func(result, data)
                    return result

                return wrapper

            return decorator

    class MockScenario:
        """Mock scenario for testing."""

        def __init__(
            self,
            name: str,
            data: Mapping[
                str,
                t.Container | MutableMapping[str, t.Container] | MutableSequence[str],
            ],
        ) -> None:
            super().__init__()
            self.name = name
            given_data = data.get("given")
            when_data = data.get("when")
            then_data = data.get("then")
            tags_data = data.get("tags")
            self.given = TestPatternsTesting.Helpers.to_general_mapping(
                given_data if isinstance(given_data, Mapping) else None,
            )
            self.when = TestPatternsTesting.Helpers.to_general_mapping(
                when_data if isinstance(when_data, Mapping) else None,
            )
            self.then = TestPatternsTesting.Helpers.to_general_mapping(
                then_data if isinstance(then_data, Mapping) else None,
            )
            self.tags = TestPatternsTesting.Helpers.to_string_list(
                tags_data
                if isinstance(tags_data, Sequence)
                and not isinstance(tags_data, str | bytes)
                else None,
            )
            priority_data = data.get("priority")
            self.priority = TestPatternsTesting.Helpers.to_string(
                priority_data
                if isinstance(priority_data, (str, int, float, bool, datetime, Path))
                else None,
                default="normal",
            )

    class GivenWhenThenBuilder:
        """Builder for given-when-then scenarios."""

        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
            self._given: MutableMapping[str, t.Container] = {}
            self._when: MutableMapping[str, t.Container] = {}
            self._then: MutableMapping[str, t.Container] = {}
            self._tags: MutableSequence[str] = []
            self._priority = "normal"

        def given(
            self,
            _description: str,
            **kwargs: t.Container,
        ) -> TestPatternsTesting.GivenWhenThenBuilder:
            self._given.update(kwargs)
            return self

        def when(
            self,
            _description: str,
            **kwargs: t.Container,
        ) -> TestPatternsTesting.GivenWhenThenBuilder:
            self._when.update(kwargs)
            return self

        def then(
            self,
            _description: str,
            **kwargs: t.Container,
        ) -> TestPatternsTesting.GivenWhenThenBuilder:
            self._then.update(kwargs)
            return self

        def with_tag(self, tag: str) -> TestPatternsTesting.GivenWhenThenBuilder:
            self._tags.append(tag)
            return self

        def with_priority(
            self,
            priority: str,
        ) -> TestPatternsTesting.GivenWhenThenBuilder:
            self._priority = priority
            return self

        def build(self) -> TestPatternsTesting.MockScenario:
            data: Mapping[
                str,
                t.Container | MutableMapping[str, t.Container] | MutableSequence[str],
            ] = {
                "given": self._given,
                "when": self._when,
                "then": self._then,
                "tags": self._tags,
                "priority": self._priority,
            }
            return TestPatternsTesting.MockScenario(self.name, data)

    class FlextTestBuilder:
        """Builder for test data."""

        def __init__(self) -> None:
            super().__init__()
            self._data: t.MutableContainerMapping = dict[str, t.NormalizedValue]()

        def with_id(self, id_: str) -> TestPatternsTesting.FlextTestBuilder:
            self._data["id"] = id_
            return self

        def with_correlation_id(
            self,
            correlation_id: str,
        ) -> TestPatternsTesting.FlextTestBuilder:
            self._data["correlation_id"] = correlation_id
            return self

        def with_metadata(
            self,
            **kwargs: t.Container,
        ) -> TestPatternsTesting.FlextTestBuilder:
            self._data.update(kwargs)
            return self

        def with_user_data(
            self,
            name: str,
            email: str,
        ) -> TestPatternsTesting.FlextTestBuilder:
            self._data["name"] = name
            self._data["email"] = email
            return self

        def with_timestamp(self) -> TestPatternsTesting.FlextTestBuilder:
            _ = self._data.setdefault("created_at", "2023-01-01T00:00:00+00:00")
            _ = self._data.setdefault("updated_at", "2023-01-01T00:00:00+00:00")
            return self

        def with_validation_rules(self) -> TestPatternsTesting.FlextTestBuilder:
            return self

        def build(self) -> FixtureDataDict:
            return self._data

    class ParameterizedTestBuilder:
        """Builder for parameterized tests."""

        def __init__(self, test_name: str) -> None:
            super().__init__()
            self.test_name = test_name
            self._cases: MutableSequence[FixtureCaseDict] = []
            self._success_cases: MutableSequence[FixtureCaseDict] = []
            self._failure_cases: MutableSequence[FixtureCaseDict] = []

        def add_case(
            self,
            email: str | None = None,
            input_value: str | None = None,
        ) -> TestPatternsTesting.ParameterizedTestBuilder:
            case: t.MutableContainerMapping = {}
            if email is not None:
                case["email"] = email
            if input_value is not None:
                case["input"] = input_value
            self._cases.append(case)
            return self

        def add_success_cases(
            self,
            cases: Sequence[FixtureCaseDict],
        ) -> TestPatternsTesting.ParameterizedTestBuilder:
            self._success_cases.extend(cases)
            return self

        def add_failure_cases(
            self,
            cases: Sequence[FixtureCaseDict],
        ) -> TestPatternsTesting.ParameterizedTestBuilder:
            self._failure_cases.extend(cases)
            return self

        def build(self) -> Sequence[FixtureCaseDict]:
            return list(self._cases)

        def build_pytest_params(self) -> Sequence[tuple[str, str, bool]]:
            success_params = [
                (str(c.get("email", "")), str(c.get("input", "")), True)
                for c in self._success_cases
            ]
            failure_params = [
                (str(c.get("email", "")), str(c.get("input", "")), False)
                for c in self._failure_cases
            ]
            return success_params + failure_params

        def build_test_ids(self) -> t.StrSequence:
            return [
                str(c.get("input", ""))
                for c in (*self._success_cases, *self._failure_cases)
            ]

    class AssertionBuilder:
        """Builder for assertions."""

        def __init__(self, data: t.NormalizedValue) -> None:
            super().__init__()
            self._data = data
            self._checks: MutableSequence[tuple[str, t.NormalizedValue]] = []

        def is_not_none(self) -> TestPatternsTesting.AssertionBuilder:
            assert self._data is not None
            return self

        def has_length(self, length: int) -> TestPatternsTesting.AssertionBuilder:
            if not isinstance(self._data, Sized):
                msg = f"Expected Sized t.NormalizedValue, got {type(self._data)}"
                raise TypeError(msg)
            assert len(self._data) == length
            return self

        def contains(
            self,
            item: t.NormalizedValue,
        ) -> TestPatternsTesting.AssertionBuilder:
            data = self._data
            if TestPatternsTesting.Helpers.object_mapping(data):
                assert item in data
                return self
            if isinstance(data, (list, tuple, set)):
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
            msg = f"Expected Container t.NormalizedValue, got {type(data)}"
            raise TypeError(msg)

        def satisfies(
            self,
            predicate: Callable[[t.NormalizedValue], bool],
            message: str = "",
        ) -> TestPatternsTesting.AssertionBuilder:
            assert predicate(self._data), message
            return self

        def assert_all(self) -> None:
            return

    class SuiteBuilder:
        """Builder for test suites."""

        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
            self._scenarios: MutableSequence[TestPatternsTesting.MockScenario] = []
            self._setup_data: t.MutableContainerMapping = dict[str, t.NormalizedValue]()
            self._tags: MutableSequence[str] = []

        def add_scenarios(
            self,
            scenarios: Sequence[TestPatternsTesting.MockScenario],
        ) -> TestPatternsTesting.SuiteBuilder:
            self._scenarios.extend(scenarios)
            return self

        def with_setup_data(
            self,
            **kwargs: t.NormalizedValue,
        ) -> TestPatternsTesting.SuiteBuilder:
            self._setup_data.update(kwargs)
            return self

        def with_tag(self, tag: str) -> TestPatternsTesting.SuiteBuilder:
            self._tags.append(tag)
            return self

        def build(self) -> FixtureSuiteDict:
            tags: t.ContainerList = list(self._tags)
            return {
                "suite_name": self.name,
                "scenario_count": len(self._scenarios),
                "tags": tags,
                "setup_data": self._setup_data,
            }

    class FixtureBuilder:
        """Builder for fixtures."""

        def __init__(self) -> None:
            super().__init__()
            self._fixtures: t.MutableContainerMapping = dict[str, t.NormalizedValue]()
            self._setups: MutableSequence[Callable[[], None]] = []
            self._teardowns: MutableSequence[Callable[[], None]] = []

        def with_user(
            self,
            **kwargs: t.Container,
        ) -> TestPatternsTesting.FixtureBuilder:
            self._fixtures["user"] = kwargs
            return self

        def with_request(
            self,
            **kwargs: t.Container,
        ) -> TestPatternsTesting.FixtureBuilder:
            self._fixtures["request"] = kwargs
            return self

        def build(self) -> FixtureFixturesDict:
            return dict(self._fixtures)

        def add_setup(
            self,
            func: Callable[[], None],
        ) -> TestPatternsTesting.FixtureBuilder:
            self._setups.append(func)
            return self

        def add_teardown(
            self,
            func: Callable[[], None],
        ) -> TestPatternsTesting.FixtureBuilder:
            self._teardowns.append(func)
            return self

        def add_fixture(
            self,
            key: str,
            value: t.Container,
        ) -> TestPatternsTesting.FixtureBuilder:
            self._fixtures[key] = value
            return self

        def setup_context(
            self,
        ) -> Callable[
            [],
            ContextManager[FixtureFixturesDict],
        ]:
            @contextmanager
            def _ctx() -> Generator[FixtureFixturesDict]:
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

    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(st.emails())
    def test_email_property_based(self, email: str) -> None:
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
    def test_user_profile_property_based(self, profile: t.StrMapping) -> None:
        assert "id" in profile
        assert "name" in profile
        assert "email" in profile
        assert isinstance(profile["id"], str)
        assert isinstance(profile["name"], str)
        assert isinstance(profile["email"], str)

    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(st.text(min_size=10, max_size=100))
    def test_string_performance_property_based(self, large_string: str) -> None:
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
        result = unicode_text.encode("utf-8").decode("utf-8")
        assert len(result) >= 0

    def test_stress_testing_load(self) -> None:
        def simple_operation() -> str:
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
        def memory_operation() -> Sequence[int]:
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
        _ = gc.collect()
        large_list = list(range(10000))
        filtered_list = list(u.filter(large_list, lambda x: x % 2 == 0))
        sorted_list = sorted(filtered_list, reverse=True)
        assert len(large_list) == 10000
        assert len(filtered_list) == 5000
        assert sorted_list[0] > sorted_list[-1]

    def test_given_when_then_pattern(self) -> None:
        scenario = (
            self
            .GivenWhenThenBuilder("user_registration")
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
        builder = (
            self
            .FlextTestBuilder()
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
        param_builder = self.ParameterizedTestBuilder("email_validation")
        _ = param_builder.add_success_cases([
            {"email": "test@example.com", "input": "valid_email_1"},
            {"email": "user@domain.org", "input": "valid_email_2"},
        ])
        _ = param_builder.add_failure_cases([
            {"email": "invalid-email", "input": "invalid_email_1"},
            {"email": "@domain.com", "input": "invalid_email_2"},
        ])
        params = param_builder.build_pytest_params()
        test_ids = param_builder.build_test_ids()
        assert len(params) == 4
        assert len(test_ids) == 4
        assert all(len(param) == 3 for param in params)

    def test_assertion_builder(self) -> None:
        test_data: t.StrSequence = ["apple", "banana", "cherry"]

        def check_all_strings(x: t.NormalizedValue) -> bool:
            values = self.Helpers.as_object_dict({"items": x}).get("items")
            values_list = self.Helpers.as_object_list(values)
            if values_list is None:
                return False
            return all(isinstance(item, str) for item in values_list)

        self.AssertionBuilder(test_data).is_not_none().has_length(3).contains(
            "banana",
        ).satisfies(check_all_strings, "all items should be strings").assert_all()

    @Helpers.mark_test_pattern("arrange_act_assert")
    def test_arrange_act_assert_decorator(self) -> None:
        def arrange_data() -> t.NormalizedValue:
            numbers: t.ContainerList = [1, 2, 3, 4, 5]
            return {"numbers": numbers}

        def act_on_data(data: t.NormalizedValue) -> t.NormalizedValue:
            payload = self.Helpers.as_object_dict(data)
            if "numbers" in payload:
                numbers = self.Helpers.as_int_list(payload["numbers"])
                if numbers is not None:
                    typed_numbers: Sequence[int] = numbers
                    return sum(typed_numbers)
            return 0

        def assert_result(
            result: t.NormalizedValue,
            original_data: t.NormalizedValue,
        ) -> None:
            assert result == 15
            payload = self.Helpers.as_object_dict(original_data)
            if "numbers" in payload:
                numbers = self.Helpers.as_int_list(payload["numbers"])
                if numbers is not None:
                    assert len(numbers) == 5

        @self.Helpers.arrange_act_assert(arrange_data, act_on_data, assert_result)
        def test_sum_calculation() -> None:
            msg = "Must use unified test helpers per Rule 3.6"
            raise NotImplementedError(msg)

        result = test_sum_calculation()
        assert result == 15

    def test_complete_test_suite_builder(self) -> None:
        scenarios: MutableSequence[TestPatternsTesting.MockScenario] = []
        scenario1 = (
            self
            .GivenWhenThenBuilder("successful_operation")
            .given("valid input data", data_valid=True)
            .when("operation is executed", executed=True)
            .then("operation succeeds", success=True)
            .with_tag("success")
            .build()
        )
        scenarios.append(scenario1)
        scenario2 = (
            self
            .GivenWhenThenBuilder("failed_operation")
            .given("invalid input data", data_valid=False)
            .when("operation is executed", executed=True)
            .then("operation fails gracefully", success=False, graceful=True)
            .with_tag("failure")
            .build()
        )
        scenarios.append(scenario2)
        scenario_list: Sequence[TestPatternsTesting.MockScenario] = scenarios
        suite = (
            self
            .SuiteBuilder("comprehensive_operation_tests")
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
        empty_setup: t.MutableContainerMapping = {}
        typed_setup_data = (
            {str(key): item for key, item in setup_data.items()}
            if isinstance(setup_data, dict)
            else empty_setup
        )
        if "environment" in typed_setup_data:
            env_value = typed_setup_data["environment"]
            assert env_value == "test"

    def test_api_request_processing(self) -> None:
        fixture_builder = self.FixtureBuilder()

        def setup_api_environment() -> None:
            pass

        def teardown_api_environment() -> None:
            pass

        _ = fixture_builder.add_setup(setup_api_environment)
        _ = fixture_builder.add_teardown(teardown_api_environment)
        _ = fixture_builder.add_fixture("api_base_url", "https://api.test.com")
        _ = fixture_builder.add_fixture("timeout", 30)
        with fixture_builder.setup_context()():
            test_request: t.ContainerMapping = {
                "method": "POST",
                "url": "https://api.example.com/users",
                "correlation_id": "corr_12345678",
                "headers": {"Content-Type": "application/json"},
                "body": {"name": "test"},
            }

            def process_api_request(
                request: t.ContainerMapping,
            ) -> t.ContainerMapping:
                return {
                    "status": "success",
                    "method": request["method"],
                    "url": request["url"],
                    "correlation_id": request["correlation_id"],
                    "processed_at": time.time(),
                }

            result = process_api_request(test_request)

            def check_status_success(x: t.NormalizedValue) -> bool:
                payload = self.Helpers.as_object_dict(x)
                return payload.get("status") == "success"

            def check_correlation_id(x: t.NormalizedValue) -> bool:
                payload = self.Helpers.as_object_dict(x)
                return "correlation_id" in payload

            def check_valid_method(x: t.NormalizedValue) -> bool:
                payload = self.Helpers.as_object_dict(x)
                method = payload.get("method")
                return isinstance(method, str) and method in {
                    "GET",
                    "POST",
                    "PUT",
                    "DELETE",
                    "PATCH",
                }

            self.AssertionBuilder(result).is_not_none().satisfies(
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
        config: Mapping[str, t.Container] = {
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
            self
            .GivenWhenThenBuilder("configuration_validation")
            .given(
                "a configuration t.NormalizedValue",
                config_environment=str(config["environment"]),
            )
            .when("configuration is validated", action="validate")
            .then("all required fields are present", validated=True)
            .with_tag("configuration")
            .build()
        )
        assert scenario.given.get("config_environment") == config["environment"]
        assert "configuration" in scenario.tags
