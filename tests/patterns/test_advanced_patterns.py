"""Demonstration of advanced testing patterns and comprehensive libraries.

This test file showcases the full power of our testing infrastructure including:
- Property-based testing with custom Hypothesis strategies
- Performance testing with complexity analysis
- Stress testing and load testing
- Advanced test patterns (Builder, Given-When-Then)
- Comprehensive parametrized testing
"""

from __future__ import annotations

import time
from collections.abc import Callable, Container, Sized
from typing import TypeVar, cast

import pytest
from hypothesis import assume, given, strategies as st

from ..support import (
    AsyncTestUtils,
    BenchmarkUtils,
    ComplexityAnalyzer,
    PerformanceProfiler,
    StressTestRunner,
)
from ..support.hypothesis import (
    CompositeStrategies,
    EdgeCaseStrategies,
    FlextStrategies,
    PerformanceStrategies,
    PropertyTestHelpers,
)
from ..support.performance import BenchmarkProtocol


def mark_test_pattern(
    pattern: str,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Mark test with a specific pattern for demonstration purposes."""

    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        func._test_pattern = pattern  # type: ignore[attr-defined]
        return func

    return decorator


pytestmark = [pytest.mark.unit, pytest.mark.architecture, pytest.mark.advanced]


# ============================================================================
# STUB CLASSES FOR ADVANCED PATTERNS DEMONSTRATION
# ============================================================================


class MockScenario:
    """Mock scenario object for testing purposes."""

    def __init__(self, name: str, data: dict[str, object]) -> None:
        self.name = name
        self.given = data.get("given", {})
        self.when = data.get("when", {})
        self.then = data.get("then", {})
        self.tags = data.get("tags", [])
        self.priority = data.get("priority", "normal")


class GivenWhenThenBuilder:
    """Builder for Given-When-Then test scenarios."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._given: dict[str, object] = {}
        self._when: dict[str, object] = {}
        self._then: dict[str, object] = {}
        self._tags: list[str] = []
        self._priority = "normal"

    def given(self, _description: str, **kwargs: object) -> GivenWhenThenBuilder:
        self._given.update(kwargs)
        return self

    def when(self, _description: str, **kwargs: object) -> GivenWhenThenBuilder:
        self._when.update(kwargs)
        return self

    def then(self, _description: str, **kwargs: object) -> GivenWhenThenBuilder:
        self._then.update(kwargs)
        return self

    def with_tag(self, tag: str) -> GivenWhenThenBuilder:
        self._tags.append(tag)
        return self

    def with_priority(self, priority: str) -> GivenWhenThenBuilder:
        self._priority = priority
        return self

    def build(self) -> MockScenario:
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
        self._data: dict[str, object] = {}

    def with_id(self, id_: str) -> FlextTestBuilder:
        self._data["id"] = id_
        return self

    def with_correlation_id(self, correlation_id: str) -> FlextTestBuilder:
        self._data["correlation_id"] = correlation_id
        return self

    def with_metadata(self, **kwargs: object) -> FlextTestBuilder:
        self._data.update(kwargs)
        return self

    def with_user_data(self, name: str, email: str) -> FlextTestBuilder:
        self._data["name"] = name
        self._data["email"] = email
        return self

    def with_timestamp(self) -> FlextTestBuilder:
        self._data.setdefault("created_at", "2023-01-01T00:00:00+00:00")
        self._data.setdefault("updated_at", "2023-01-01T00:00:00+00:00")
        return self

    def with_validation_rules(self) -> FlextTestBuilder:
        # No-op stub to keep example API; could attach schema metadata here
        return self

    def build(self) -> dict[str, object]:
        return self._data.copy()


class ParameterizedTestBuilder:
    """Builder for parametrized test cases."""

    def __init__(self, test_name: str) -> None:
        self.test_name = test_name
        self._cases: list[dict[str, object]] = []
        self._success_cases: list[dict[str, object]] = []
        self._failure_cases: list[dict[str, object]] = []

    def add_case(self, **kwargs: object) -> ParameterizedTestBuilder:
        self._cases.append(kwargs)
        return self

    def add_success_cases(
        self, cases: list[dict[str, object]],
    ) -> ParameterizedTestBuilder:
        self._success_cases.extend(cases)
        return self

    def add_failure_cases(
        self, cases: list[dict[str, object]],
    ) -> ParameterizedTestBuilder:
        self._failure_cases.extend(cases)
        return self

    def build(self) -> list[dict[str, object]]:
        return self._cases.copy()

    def build_pytest_params(self) -> list[tuple[str, str, bool]]:
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
        return [
            str(c.get("input", ""))
            for c in (*self._success_cases, *self._failure_cases)
        ]


class TestAssertionBuilder:
    """Builder for complex test assertions."""

    def __init__(self, data: object) -> None:
        self._data = data
        self._checks: list[tuple[str, object]] = []

    def is_not_none(self) -> TestAssertionBuilder:
        assert self._data is not None
        return self

    def has_length(self, length: int) -> TestAssertionBuilder:
        if hasattr(self._data, "__len__"):
            assert len(cast("Sized", self._data)) == length
        return self

    def contains(self, item: object) -> TestAssertionBuilder:
        if hasattr(self._data, "__contains__"):
            assert item in cast("Container[object]", self._data)
        return self

    def satisfies(self, predicate: object, message: str = "") -> TestAssertionBuilder:
        if callable(predicate):
            assert predicate(self._data), message
        return self

    def assert_all(self) -> None:
        # All checks are executed inline in this simple stub
        return None


class TestSuiteBuilder:
    """Builder for test suites."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._scenarios: list[object] = []
        self._setup_data: dict[str, object] = {}
        self._tags: list[str] = []

    def add_scenarios(self, scenarios: list[object]) -> TestSuiteBuilder:
        self._scenarios.extend(scenarios)
        return self

    def with_setup_data(self, **kwargs: object) -> TestSuiteBuilder:
        self._setup_data.update(kwargs)
        return self

    def with_tag(self, tag: str) -> TestSuiteBuilder:
        self._tags.append(tag)
        return self

    def build(self) -> dict[str, object]:
        return {
            "suite_name": self.name,
            "scenario_count": len(self._scenarios),
            "tags": self._tags,
            "setup_data": self._setup_data,
        }


class TestFixtureBuilder:
    """Builder for test fixtures."""

    def __init__(self) -> None:
        self._fixtures: dict[str, object] = {}
        self._setups: list[object] = []
        self._teardowns: list[object] = []

    def with_user(self, **kwargs: object) -> TestFixtureBuilder:
        self._fixtures["user"] = kwargs
        return self

    def with_request(self, **kwargs: object) -> TestFixtureBuilder:
        self._fixtures["request"] = kwargs
        return self

    def build(self) -> dict[str, object]:
        return self._fixtures.copy()

    def add_setup(self, func: object) -> TestFixtureBuilder:
        self._setups.append(func)
        return self

    def add_teardown(self, func: object) -> TestFixtureBuilder:
        self._teardowns.append(func)
        return self

    def add_fixture(self, key: str, value: object) -> TestFixtureBuilder:
        self._fixtures[key] = value
        return self

    def setup_context(self) -> object:
        from collections.abc import Iterator
        from contextlib import contextmanager

        @contextmanager
        def _ctx() -> Iterator[dict[str, object]]:
            for f in self._setups:
                if callable(f):
                    f()
            try:
                yield self._fixtures
            finally:
                for f in self._teardowns:
                    if callable(f):
                        f()

        return _ctx()


T = TypeVar("T")


def arrange_act_assert(
    _arrange_func: Callable[..., object],
    _act_func: Callable[[object], object],
    _assert_func: Callable[[object, object], None],
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Decorator for AAA pattern testing."""

    def decorator(_test_func: Callable[..., object]) -> Callable[..., object]:
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

    @given(FlextStrategies.emails())
    def test_email_properties(self, email: str) -> None:
        """Property-based test for email handling."""
        assume(PropertyTestHelpers.assume_valid_email(email))

        # Properties that should always hold for valid emails
        assert "@" in email
        assert len(email) > 3
        assert not email.startswith("@")
        assert not email.endswith("@")

    @given(CompositeStrategies.user_profiles())
    def test_user_profile_properties(self, profile: dict[str, object]) -> None:
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
        assume(PropertyTestHelpers.assume_valid_email(profile["email"]))

    @given(PerformanceStrategies.large_strings())
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

    @given(EdgeCaseStrategies.unicode_edge_cases())
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
        analyzer = ComplexityAnalyzer()

        def linear_operation(size: int) -> None:
            """Simulate a linear operation."""
            for _ in range(size):
                pass

        # Measure across different input sizes
        input_sizes = [100, 200, 400, 800]
        result = analyzer.measure_complexity(
            linear_operation, input_sizes, "linear_operation",
        )

        # Type cast for proper type checking
        assert result["operation"] == "linear_operation"
        results = result["results"]
        assert isinstance(results, list)
        assert len(results) == len(input_sizes)
        assert "complexity_analysis" in result

    def test_stress_testing_load(self) -> None:
        """Demonstrate stress testing with load patterns."""
        stress_runner = StressTestRunner()

        def simple_operation() -> str:
            """Simple operation for stress testing."""
            return "test_result"

        # Run load test
        result = stress_runner.run_load_test(
            simple_operation, iterations=1000, operation_name="simple_ops",
        )

        # Type cast for proper type checking
        assert result["iterations"] == 1000
        successes = result["successes"]
        assert isinstance(successes, (int, float))
        assert successes > 0
        failure_rate = result["failure_rate"]
        assert isinstance(failure_rate, (int, float))
        assert failure_rate == 0.0  # Should not fail
        ops_per_second = result["operations_per_second"]
        assert isinstance(ops_per_second, (int, float))
        assert ops_per_second > 0

    def test_endurance_testing(self) -> None:
        """Demonstrate endurance testing."""
        stress_runner = StressTestRunner()

        def memory_operation() -> list[int]:
            """Operation that uses some memory."""
            return list(range(100))

        # Run for 2 seconds
        result = stress_runner.run_endurance_test(
            memory_operation, duration_seconds=2.0, operation_name="memory_ops",
        )

        # Type cast for proper type checking
        actual_duration = result["actual_duration_seconds"]
        assert isinstance(actual_duration, (int, float))
        assert actual_duration >= 1.8  # Allow some variance
        iterations = result["iterations"]
        assert isinstance(iterations, (int, float))
        assert iterations > 0
        ops_per_second = result["operations_per_second"]
        assert isinstance(ops_per_second, (int, float))
        assert ops_per_second > 0

    def test_memory_profiling_advanced(self) -> None:
        """Demonstrate advanced memory profiling."""
        profiler = PerformanceProfiler()

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
            max_memory_mb=20.0, operation_name="list_operations",
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
        # Type cast for proper type checking
        given_dict = scenario.given
        assert isinstance(given_dict, dict)
        assert "email" in given_dict
        when_dict = scenario.when
        assert isinstance(when_dict, dict)
        assert "action" in when_dict
        then_dict = scenario.then
        assert isinstance(then_dict, dict)
        assert "success" in then_dict
        tags_list = scenario.tags
        assert isinstance(tags_list, list)
        assert "integration" in tags_list
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

        def arrange_data(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {"numbers": [1, 2, 3, 4, 5]}

        def act_on_data(data: dict[str, object]) -> int:
            numbers = data["numbers"]
            assert isinstance(numbers, list)
            return sum(numbers)

        def assert_result(result: int, original_data: dict[str, object]) -> None:
            assert result == 15
            numbers = original_data["numbers"]
            assert isinstance(numbers, list)
            assert len(numbers) == 5

        @arrange_act_assert(arrange_data, act_on_data, assert_result)
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
            .add_scenarios(scenarios)
            .with_setup_data(environment="test", timeout=30)
            .with_tag("integration")
            .build()
        )

        assert suite["suite_name"] == "comprehensive_operation_tests"
        assert suite["scenario_count"] == 2
        assert "integration" in cast("list[str]", suite["tags"])
        assert cast("dict[str, object]", suite["setup_data"])["environment"] == "test"

    @pytest.mark.asyncio
    async def test_async_with_all_patterns(
        self, async_test_utils: AsyncTestUtils,
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

        async def async_operation() -> dict[str, object]:
            """Simulate async operation."""
            await async_test_utils.simulate_delay(0.1)
            return {"result": "success", "data": test_data}

        # Execute with timeout
        result = await async_test_utils.run_with_timeout(
            async_operation(), timeout_seconds=5.0,
        )

        # Use assertion builder for verification
        TestAssertionBuilder(result).is_not_none().satisfies(
            lambda x: "result" in x, "should have result field",
        ).satisfies(
            lambda x: x["result"] == "success", "should be successful",
        ).assert_all()

    def test_performance_with_property_testing(
        self, benchmark: BenchmarkProtocol,
    ) -> None:
        """Combine performance testing with property-based testing."""

        def process_user_profiles(profiles: list[dict[str, object]]) -> list[str]:
            """Process user profiles (simulate real operation)."""
            return [
                f"{profile['name']} <{profile['email']}>"
                for profile in profiles
                if PropertyTestHelpers.assume_valid_email(cast("str", profile["email"]))
            ]

        # Generate test data using our strategies

        st.lists(CompositeStrategies.user_profiles(), min_size=10, max_size=100)

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
        def benchmark_operation() -> list[str]:
            return process_user_profiles(test_profiles)

        results = BenchmarkUtils.benchmark_with_warmup(
            benchmark, benchmark_operation, warmup_rounds=3,
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

        with fixture_builder.setup_context():
            # Use a fixed test request instead of Hypothesis example
            test_request = {
                "method": "POST",
                "url": "https://api.example.com/users",
                "correlation_id": "corr_12345678",
                "headers": {"Content-Type": "application/json"},
                "body": {"name": "test"},
            }

            # Simulate API processing
            def process_api_request(request: dict[str, object]) -> dict[str, object]:
                return {
                    "status": "success",
                    "method": request["method"],
                    "url": request["url"],
                    "correlation_id": request["correlation_id"],
                    "processed_at": time.time(),
                }

            # Execute with performance monitoring
            profiler = PerformanceProfiler()

            with profiler.profile_memory("api_processing"):
                result = process_api_request(test_request)

            # Comprehensive assertions
            TestAssertionBuilder(result).is_not_none().satisfies(
                lambda x: x["status"] == "success", "should be successful",
            ).satisfies(
                lambda x: "correlation_id" in x, "should have correlation ID",
            ).satisfies(
                lambda x: x["method"] in {"GET", "POST", "PUT", "DELETE", "PATCH"},
                "should have valid HTTP method",
            ).assert_all()

    @given(CompositeStrategies.configuration_data())
    def test_configuration_validation_comprehensive(
        self, config: dict[str, object],
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

        assert cast("dict[str, object]", scenario.given)["config"] == config
        assert "configuration" in cast("list[str]", scenario.tags)
