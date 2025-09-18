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

import asyncio
import time
from collections.abc import Callable, Container, Iterator, Sized
from contextlib import AbstractContextManager as ContextManager, contextmanager
from typing import cast

import pytest
from hypothesis import assume, given, strategies as st

from flext_core import FlextTypes, T
from flext_tests import (
    FlextTestsAsyncs,
    FlextTestsHypothesis,
    FlextTestsPerformance,
)


def mark_test_pattern(
    pattern: str,
) -> Callable[[Callable[[T], None]], Callable[[T], None]]:
    """Mark test with a specific pattern for demonstration purposes."""

    def decorator(func: Callable[[T], None]) -> Callable[[T], None]:
        # Use hasattr/setattr for dynamic attribute setting to avoid PyRight error
        setattr(func, "_test_pattern", pattern)
        return func

    return decorator


pytestmark = [pytest.mark.unit, pytest.mark.architecture, pytest.mark.advanced]


# ============================================================================
# STUB CLASSES FOR ADVANCED PATTERNS DEMONSTRATION
# ============================================================================


class MockScenario:
    """Mock scenario object for testing purposes."""

    def __init__(self, name: str, data: FlextTypes.Core.Dict) -> None:
        """Initialize mock scenario with name and test data."""
        self.name = name
        self.given = data.get("given", {})
        self.when = data.get("when", {})
        self.then = data.get("then", {})
        self.tags = data.get("tags", [])
        self.priority = data.get("priority", "normal")


class GivenWhenThenBuilder:
    """Builder for Given-When-Then test scenarios."""

    def __init__(self, name: str) -> None:
        """Initialize Given-When-Then builder with test name."""
        self.name = name
        self._given: FlextTypes.Core.Dict = {}
        self._when: FlextTypes.Core.Dict = {}
        self._then: FlextTypes.Core.Dict = {}
        self._tags: FlextTypes.Core.StringList = []
        self._priority = "normal"

    def given(self, _description: str, **kwargs: object) -> GivenWhenThenBuilder:
        """Add given conditions to the test scenario."""
        self._given.update(kwargs)
        return self

    def when(self, _description: str, **kwargs: object) -> GivenWhenThenBuilder:
        """Add when actions to the test scenario."""
        self._when.update(kwargs)
        return self

    def then(self, _description: str, **kwargs: object) -> GivenWhenThenBuilder:
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
    """Builder for test data objects."""

    def __init__(self) -> None:
        """Initialize test data builder with empty data."""
        self._data: FlextTypes.Core.Dict = {}

    def with_id(self, id_: str) -> FlextTestBuilder:
        """Add ID to the test data."""
        self._data["id"] = id_
        return self

    def with_correlation_id(self, correlation_id: str) -> FlextTestBuilder:
        """Add correlation ID to the test data."""
        self._data["correlation_id"] = correlation_id
        return self

    def with_metadata(self, **kwargs: object) -> FlextTestBuilder:
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
        self._data.setdefault("created_at", "2023-01-01T00:00:00+00:00")
        self._data.setdefault("updated_at", "2023-01-01T00:00:00+00:00")
        return self

    def with_validation_rules(self) -> FlextTestBuilder:
        """Add validation rules to the test data (no-op stub)."""
        # No-op stub to keep example API; could attach schema metadata here
        return self

    def build(self) -> FlextTypes.Core.Dict:
        """Build the final test data dictionary."""
        return self._data.copy()


class ParameterizedTestBuilder:
    """Builder for parametrized test cases."""

    def __init__(self, test_name: str) -> None:
        """Initialize parameterized test builder with test name."""
        self.test_name = test_name
        self._cases: list[FlextTypes.Core.Dict] = []
        self._success_cases: list[FlextTypes.Core.Dict] = []
        self._failure_cases: list[FlextTypes.Core.Dict] = []

    def add_case(self, **kwargs: object) -> ParameterizedTestBuilder:
        """Add a test case with the given parameters."""
        self._cases.append(kwargs)
        return self

    def add_success_cases(
        self,
        cases: list[FlextTypes.Core.Dict],
    ) -> ParameterizedTestBuilder:
        """Add multiple success test cases."""
        self._success_cases.extend(cases)
        return self

    def add_failure_cases(
        self,
        cases: list[FlextTypes.Core.Dict],
    ) -> ParameterizedTestBuilder:
        """Add multiple failure test cases."""
        self._failure_cases.extend(cases)
        return self

    def build(self) -> list[FlextTypes.Core.Dict]:
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

    def build_test_ids(self) -> FlextTypes.Core.StringList:
        """Build test IDs for pytest parametrization."""
        return [
            str(c.get("input", ""))
            for c in (*self._success_cases, *self._failure_cases)
        ]


class TestAssertionBuilder:
    """Builder for complex test assertions."""

    def __init__(self, data: object) -> None:
        """Initialize test assertion builder with data to test."""
        self._data = data
        self._checks: list[tuple[str, object]] = []

    def is_not_none(self) -> TestAssertionBuilder:
        """Assert that the data is not None."""
        assert self._data is not None
        return self

    def has_length(self, length: int) -> TestAssertionBuilder:
        """Assert that the data has the specified length."""
        assert len(cast("Sized", self._data)) == length
        return self

    def contains(self, item: object) -> TestAssertionBuilder:
        """Assert that the data contains the specified item."""
        assert item in cast("Container[object]", self._data)
        return self

    def satisfies(self, predicate: object, message: str = "") -> TestAssertionBuilder:
        """Assert that the data satisfies the given predicate."""
        if callable(predicate):
            assert predicate(self._data), message
        return self

    def assert_all(self) -> None:
        """Execute all accumulated assertions."""
        # All checks are executed inline in this simple stub
        return


class TestSuiteBuilder:
    """Builder for test suites."""

    def __init__(self, name: str) -> None:
        """Initialize test suite builder with suite name."""
        self.name = name
        self._scenarios: FlextTypes.Core.List = []
        self._setup_data: FlextTypes.Core.Dict = {}
        self._tags: FlextTypes.Core.StringList = []

    def add_scenarios(self, scenarios: FlextTypes.Core.List) -> TestSuiteBuilder:
        """Add multiple test scenarios to the suite."""
        self._scenarios.extend(scenarios)
        return self

    def with_setup_data(self, **kwargs: object) -> TestSuiteBuilder:
        """Add setup data to the test suite."""
        self._setup_data.update(kwargs)
        return self

    def with_tag(self, tag: str) -> TestSuiteBuilder:
        """Add a tag to the test suite."""
        self._tags.append(tag)
        return self

    def build(self) -> FlextTypes.Core.Dict:
        """Build the test suite configuration."""
        return {
            "suite_name": self.name,
            "scenario_count": len(self._scenarios),
            "tags": self._tags,
            "setup_data": self._setup_data,
        }


class TestFixtureBuilder:
    """Builder for test fixtures."""

    def __init__(self) -> None:
        """Initialize test fixture builder with empty fixtures."""
        self._fixtures: FlextTypes.Core.Dict = {}
        self._setups: FlextTypes.Core.List = []
        self._teardowns: FlextTypes.Core.List = []

    def with_user(self, **kwargs: object) -> TestFixtureBuilder:
        """Add user fixture data."""
        self._fixtures["user"] = kwargs
        return self

    def with_request(self, **kwargs: object) -> TestFixtureBuilder:
        """Add request fixture data."""
        self._fixtures["request"] = kwargs
        return self

    def build(self) -> FlextTypes.Core.Dict:
        """Build the test fixtures configuration."""
        return self._fixtures.copy()

    def add_setup(self, func: object) -> TestFixtureBuilder:
        """Add a setup function to the fixtures."""
        self._setups.append(func)
        return self

    def add_teardown(self, func: object) -> TestFixtureBuilder:
        """Add a teardown function to the fixtures."""
        self._teardowns.append(func)
        return self

    def add_fixture(self, key: str, value: object) -> TestFixtureBuilder:
        """Add a custom fixture with the given key and value."""
        self._fixtures[key] = value
        return self

    def setup_context(self) -> Callable[[], ContextManager[FlextTypes.Core.Dict]]:
        """Create a context manager for test setup and teardown."""

        @contextmanager
        def _ctx() -> Iterator[FlextTypes.Core.Dict]:
            for f in self._setups:
                if callable(f):
                    f()
            try:
                yield self._fixtures
            finally:
                for f in self._teardowns:
                    if callable(f):
                        f()

        return _ctx


def arrange_act_assert(
    _arrange_func: Callable[[], object],
    _act_func: Callable[[object], object],
    _assert_func: Callable[[object, object], None],
) -> Callable[[Callable[[], object]], Callable[[], object]]:
    """Decorator for AAA pattern testing."""

    def decorator(_test_func: Callable[[], object]) -> Callable[[], object]:
        def wrapper() -> object:
            data = _arrange_func()
            result = _act_func(data)
            _assert_func(result, data)
            return result

        return wrapper

    return decorator


# ============================================================================
# PROPERTY-BASED TESTING DEMONSTRATIONS
# ============================================================================


class TestPropertyBasedPatterns:
    """Demonstrate property-based testing with custom strategies."""

    @given(FlextTestsHypothesis.FlextStrategies.emails())
    def test_email_properties(self, email: str) -> None:
        """Property-based test for email handling."""
        assume(FlextTestsHypothesis.PropertyTestHelpers.assume_valid_email(email))

        # Properties that should always hold for valid emails
        assert "@" in email
        assert len(email) > 3
        assert not email.startswith("@")
        assert not email.endswith("@")

    @given(FlextTestsHypothesis.CompositeStrategies.user_profiles())
    def test_user_profile_properties(self, profile: FlextTypes.Core.Dict) -> None:
        """Property-based test for user profiles."""
        # Verify required fields
        assert "id" in profile
        assert "name" in profile
        assert "email" in profile

        # Verify data types
        assert isinstance(profile["id"], str)
        assert isinstance(profile["name"], str)
        assert isinstance(profile["email"], str)

        # Assume valid email format (filters invalid inputs)
        assume(
            FlextTestsHypothesis.PropertyTestHelpers.assume_valid_email(
                profile["email"]
            )
        )

    @given(FlextTestsHypothesis.PerformanceStrategies.large_strings())
    def test_string_processing_scalability(self, large_string: str) -> None:
        """Property-based test for string processing performance."""
        assume(len(large_string) >= 1000)

        # Measure basic operations
        start_time = time.perf_counter()

        # Simulate string processing
        processed = large_string.lower().strip()
        word_count = len(processed.split())

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Performance properties
        assert duration < 1.0  # Should complete within 1 second
        assert word_count >= 0
        assert len(processed) <= len(large_string)

    @given(FlextTestsHypothesis.EdgeCaseStrategies.unicode_edge_cases())
    def test_unicode_handling_properties(self, unicode_text: str) -> None:
        """Property-based test for Unicode handling."""
        # Should handle Unicode gracefully without crashing
        result = unicode_text.encode("utf-8").decode("utf-8")
        assert isinstance(result, str)

        # Length might differ due to Unicode normalization
        assert len(result) >= 0


# ============================================================================
# PERFORMANCE AND COMPLEXITY ANALYSIS
# ============================================================================


class TestPerformanceAnalysis:
    """Demonstrate performance testing and complexity analysis."""

    def test_complexity_analysis_linear(self) -> None:
        """Test complexity analysis for linear algorithms."""
        analyzer = FlextTestsPerformance.ComplexityAnalyzer()

        def linear_operation(size: int) -> None:
            """Simulate a linear operation."""
            for _ in range(size):
                pass

        # Measure across different input sizes
        input_sizes = [100, 200, 400, 800]
        result = analyzer.measure_complexity(
            linear_operation,
            input_sizes,
            "linear_operation",
        )

        assert result["operation"] == "linear_operation"
        assert len(cast("Sized", result["results"])) == len(input_sizes)
        assert "complexity_analysis" in result

    def test_stress_testing_load(self) -> None:
        """Demonstrate stress testing with load patterns."""
        stress_runner = FlextTestsPerformance.StressTestRunner()

        def simple_operation() -> str:
            """Simple operation for stress testing."""
            return "test_result"

        # Run load test
        result = stress_runner.run_load_test(
            simple_operation,
            iterations=1000,
            operation_name="simple_ops",
        )

        assert result["iterations"] == 1000
        assert cast("int", result["successes"]) > 0
        assert result["failure_rate"] == 0.0  # Should not fail
        assert cast("float", result["operations_per_second"]) > 0

    def test_endurance_testing(self) -> None:
        """Demonstrate endurance testing."""
        stress_runner = FlextTestsPerformance.StressTestRunner()

        def memory_operation() -> list[int]:
            """Operation that uses some memory."""
            return list(range(100))

        # Run for 2 seconds
        result = stress_runner.run_endurance_test(
            memory_operation,
            duration_seconds=2.0,
            operation_name="memory_ops",
        )

        assert (
            cast("float", result["actual_duration_seconds"]) >= 1.8
        )  # Allow some variance
        assert cast("int", result["iterations"]) > 0
        assert cast("float", result["operations_per_second"]) > 0

    def test_memory_profiling_advanced(self) -> None:
        """Demonstrate advanced memory profiling."""
        profiler = FlextTestsPerformance.PerformanceProfiler()

        with profiler.profile_memory("list_operations"):
            # Create and manipulate large data structures
            large_list = list(range(10000))
            filtered_list = [x for x in large_list if x % 2 == 0]
            sorted_list = sorted(filtered_list, reverse=True)

            # Verify operations completed
            assert len(large_list) == 10000
            assert len(filtered_list) == 5000
            assert sorted_list[0] > sorted_list[-1]

        profiler.assert_memory_efficient(
            max_memory_mb=20.0,
            operation_name="list_operations",
        )


# ============================================================================
# ADVANCED TEST PATTERNS
# ============================================================================


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
        assert "email" in cast("FlextTypes.Core.Dict", scenario.given)
        assert "action" in cast("FlextTypes.Core.Dict", scenario.when)
        assert "success" in cast("FlextTypes.Core.Dict", scenario.then)
        assert "integration" in cast("FlextTypes.Core.StringList", scenario.tags)
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

        assert data["id"] == "test_123"
        assert data["correlation_id"] == "corr_456"
        assert data["name"] == "John Doe"
        assert data["email"] == "john@example.com"
        assert "created_at" in data
        assert "updated_at" in data

    def test_parametrized_builder(self) -> None:
        """Demonstrate parametrized test builder."""
        param_builder = ParameterizedTestBuilder("email_validation")

        # Add various test cases
        param_builder.add_success_cases(
            [
                {"email": "test@example.com", "input": "valid_email_1"},
                {"email": "user@domain.org", "input": "valid_email_2"},
            ],
        )

        param_builder.add_failure_cases(
            [
                {"email": "invalid-email", "input": "invalid_email_1"},
                {"email": "@domain.com", "input": "invalid_email_2"},
            ],
        )

        params = param_builder.build_pytest_params()
        test_ids = param_builder.build_test_ids()

        assert len(params) == 4
        assert len(test_ids) == 4
        assert all(
            len(param) == 3 for param in params
        )  # email, input, expected_success

    def test_assertion_builder(self) -> None:
        """Demonstrate assertion builder pattern."""
        test_data = ["apple", "banana", "cherry"]

        # Build complex assertions
        TestAssertionBuilder(test_data).is_not_none().has_length(3).contains(
            "banana",
        ).satisfies(
            lambda x: all(isinstance(item, str) for item in x),
            "all items should be strings",
        ).assert_all()

    @mark_test_pattern("arrange_act_assert")
    def test_arrange_act_assert_decorator(self) -> None:
        """Demonstrate Arrange-Act-Assert pattern decorator."""

        def arrange_data(*_args: object) -> FlextTypes.Core.Dict:
            return {"numbers": [1, 2, 3, 4, 5]}

        def act_on_data(data: FlextTypes.Core.Dict) -> int:
            return sum(cast("list[int]", data["numbers"]))

        def assert_result(result: int, original_data: FlextTypes.Core.Dict) -> None:
            assert result == 15
            assert len(cast("list[int]", original_data["numbers"])) == 5

        @arrange_act_assert(
            arrange_data,
            cast("Callable[[object], object]", act_on_data),
            cast("Callable[[object, object], None]", assert_result),
        )
        def test_sum_calculation() -> None:
            pass  # Logic is in the decorator

        # Execute the decorated test
        result = test_sum_calculation()
        assert result == 15


# ============================================================================
# INTEGRATION OF ALL PATTERNS
# ============================================================================


class TestComprehensiveIntegration:
    """Demonstrate integration of all testing patterns."""

    def test_complete_test_suite_builder(self) -> None:
        """Demonstrate complete test suite construction."""
        # Create multiple scenarios
        scenarios = []

        # Scenario 1: Success case
        scenario1 = (
            GivenWhenThenBuilder("successful_operation")
            .given("valid input data", data={"valid": True})
            .when("operation is executed", executed=True)
            .then("operation succeeds", success=True)
            .with_tag("success")
            .build()
        )
        scenarios.append(scenario1)

        # Scenario 2: Failure case
        scenario2 = (
            GivenWhenThenBuilder("failed_operation")
            .given("invalid input data", data={"valid": False})
            .when("operation is executed", executed=True)
            .then("operation fails gracefully", success=False, graceful=True)
            .with_tag("failure")
            .build()
        )
        scenarios.append(scenario2)

        # Build complete test suite
        suite = (
            TestSuiteBuilder("comprehensive_operation_tests")
            .add_scenarios(cast("FlextTypes.Core.List", scenarios))
            .with_setup_data(environment="test", timeout=30)
            .with_tag("integration")
            .build()
        )

        assert suite["suite_name"] == "comprehensive_operation_tests"
        assert suite["scenario_count"] == 2
        assert "integration" in cast("FlextTypes.Core.StringList", suite["tags"])
        assert (
            cast("FlextTypes.Core.Dict", suite["setup_data"])["environment"] == "test"
        )

    @pytest.mark.asyncio
    async def test_async_with_all_patterns(
        self,
        async_test_utils: FlextTestsAsyncs,
    ) -> None:
        """Demonstrate async testing with all patterns."""
        # Build test data
        test_data = (
            FlextTestBuilder()
            .with_id("async_test")
            .with_correlation_id("async_corr")
            .with_validation_rules()
            .build()
        )

        async def async_operation() -> FlextTypes.Core.Dict:
            """Simulate async operation."""
            await asyncio.sleep(0.1)
            return {"result": "success", "data": test_data}

        # Execute with timeout
        result = await async_test_utils.run_with_timeout(
            async_operation(),
            timeout_seconds=5.0,
        )

        # Use assertion builder for verification
        TestAssertionBuilder(result).is_not_none().satisfies(
            lambda x: "result" in x,
            "should have result field",
        ).satisfies(
            lambda x: x["result"] == "success",
            "should be successful",
        ).assert_all()

    def test_performance_with_property_testing(
        self,
        benchmark: FlextTestsPerformance.BenchmarkProtocol,
    ) -> None:
        """Combine performance testing with property-based testing."""

        def process_user_profiles(
            profiles: list[FlextTypes.Core.Dict],
        ) -> FlextTypes.Core.StringList:
            """Process user profiles (simulate real operation)."""
            return [
                f"{profile['name']} <{profile['email']}>"
                for profile in profiles
                if FlextTestsHypothesis.PropertyTestHelpers.assume_valid_email(
                    cast("str", profile["email"])
                )
            ]

        # Generate test data using our strategies

        st.lists(
            FlextTestsHypothesis.CompositeStrategies.user_profiles(),
            min_size=10,
            max_size=100,
        )

        # Use a fixed set of test profiles instead of Hypothesis example
        test_profiles = [
            {
                "id": "flext_test001",
                "name": "John Smith",
                "email": "john.smith@example.com",
                "phone": "+1-555-123-4567",
                "address": {
                    "street": "123 Main St",
                    "city": "Springfield",
                    "state": "CA",
                    "zip_code": "12345",
                },
                "created_at": "2023-01-01T00:00:00+00:00",
                "active": True,
                "metadata": {},
            },
            {
                "id": "flext_test002",
                "name": "Jane Doe",
                "email": "jane.doe@example.com",
                "phone": "+1-555-987-6543",
                "address": {
                    "street": "456 Oak Ave",
                    "city": "Madison",
                    "state": "NY",
                    "zip_code": "54321",
                },
                "created_at": "2023-06-15T12:30:00+00:00",
                "active": True,
                "metadata": {"role": "REDACTED_LDAP_BIND_PASSWORD"},
            },
        ]

        # Benchmark the operation
        def benchmark_operation() -> FlextTypes.Core.StringList:
            return process_user_profiles(test_profiles)

        results = FlextTestsPerformance.BenchmarkUtils.benchmark_with_warmup(
            benchmark,
            benchmark_operation,
            warmup_rounds=3,
        )

        # Verify results
        assert isinstance(results, list)
        assert all(isinstance(item, str) for item in results)
        assert all("@" in item for item in results)


# ============================================================================
# REAL-WORLD SCENARIO SIMULATION
# ============================================================================


class TestRealWorldScenarios:
    """Simulate real-world testing scenarios."""

    def test_api_request_processing(self) -> None:
        """Simulate API request processing with comprehensive testing."""
        # Create test fixtures
        fixture_builder = TestFixtureBuilder()

        # Setup
        def setup_api_environment() -> None:
            pass  # Setup API mock environment

        def teardown_api_environment() -> None:
            pass  # Cleanup

        fixture_builder.add_setup(setup_api_environment)
        fixture_builder.add_teardown(teardown_api_environment)
        fixture_builder.add_fixture("api_base_url", "https://api.test.com")
        fixture_builder.add_fixture("timeout", 30)

        with fixture_builder.setup_context()():
            # Use a fixed test request instead of Hypothesis example
            test_request: FlextTypes.Core.Dict = {
                "method": "POST",
                "url": "https://api.example.com/users",
                "correlation_id": "corr_12345678",
                "headers": {"Content-Type": "application/json"},
                "body": {"name": "test"},
            }

            # Simulate API processing
            def process_api_request(
                request: FlextTypes.Core.Dict,
            ) -> FlextTypes.Core.Dict:
                return {
                    "status": "success",
                    "method": request["method"],
                    "url": request["url"],
                    "correlation_id": request["correlation_id"],
                    "processed_at": time.time(),
                }

            # Execute with performance monitoring
            profiler = FlextTestsPerformance.PerformanceProfiler()

            with profiler.profile_memory("api_processing"):
                result = process_api_request(test_request)

            # Comprehensive assertions
            TestAssertionBuilder(result).is_not_none().satisfies(
                lambda x: x["status"] == "success",
                "should be successful",
            ).satisfies(
                lambda x: "correlation_id" in x,
                "should have correlation ID",
            ).satisfies(
                lambda x: x["method"] in {"GET", "POST", "PUT", "DELETE", "PATCH"},
                "should have valid HTTP method",
            ).assert_all()

    @given(FlextTestsHypothesis.CompositeStrategies.configuration_data())
    def test_configuration_validation_comprehensive(
        self,
        config: FlextTypes.Core.Dict,
    ) -> None:
        """Comprehensive configuration validation testing."""
        # Validate configuration structure
        required_fields = ["database_url", "debug", "timeout_seconds"]
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"

        # Validate data types
        assert isinstance(config["debug"], bool)
        assert isinstance(config["timeout_seconds"], int)
        assert config["timeout_seconds"] > 0

        # Validate environment
        assert config["environment"] in {"development", "staging", "production"}

        # Build test scenario for this configuration
        scenario = (
            GivenWhenThenBuilder("configuration_validation")
            .given("a configuration object", config=config)
            .when("configuration is validated", action="validate")
            .then("all required fields are present", validated=True)
            .with_tag("configuration")
            .build()
        )

        assert cast("FlextTypes.Core.Dict", scenario.given)["config"] == config
        assert "configuration" in cast("FlextTypes.Core.StringList", scenario.tags)
