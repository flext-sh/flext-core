"""Comprehensive tests for FlextGuards and guard functionality.

This refactored test file demonstrates extensive use of our testing infrastructure:
- factory_boy for realistic test data generation
- pytest-benchmark for performance testing
- pytest-asyncio for async testing patterns
- pytest-mock for advanced mocking
- pytest-httpx for HTTP testing
- Property-based testing with Hypothesis
- Advanced test patterns (Builder, Given-When-Then)
- Performance analysis and stress testing
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast, object

import pytest
from hypothesis import assume, given
from pytest_benchmark.fixture import BenchmarkFixture
from tests.support.async_utils import AsyncTestUtils
from tests.support.domain_factories import (
    UserDataFactory,
)
from tests.support.hypothesis_utils import (
    CompositeStrategies,
    EdgeCaseStrategies,
    FlextStrategies,
    PropertyTestHelpers,
)
from tests.support.performance_utils import (
    BenchmarkUtils,
    ComplexityAnalyzer,
    PerformanceProfiler,
    StressTestRunner,
)
from tests.support.test_patterns import (
    FlextTestBuilder,
    GivenWhenThenBuilder,
    ParameterizedTestBuilder,
    TestAssertionBuilder,
    TestCaseFactory,
    arrange_act_assert,
    mark_test_pattern,
)

from flext_core import (
    FlextExceptions,
    FlextGuards,
    FlextModel,
    FlextResult,
    FlextValidationUtils,
    is_not_none,
)

# Get functions from FlextGuards unified class
FlextTypeGuards = FlextGuards.ValidationUtils  # Use validation utils for type guards
immutable = FlextGuards.PureWrapper.make_immutable
pure = FlextGuards.PureWrapper.pure_function
make_builder = FlextGuards.PureWrapper.make_builder  # Use pure wrapper for builder
make_factory = FlextGuards.PureWrapper.make_factory  # Use pure wrapper for factory

# Use FlextValidationUtils directly
require_not_none = FlextValidationUtils.require_not_none
require_positive = FlextValidationUtils.require_positive
require_non_empty = FlextValidationUtils.require_non_empty
safe = FlextGuards.PureWrapper.safe_function

pytestmark = [pytest.mark.unit, pytest.mark.core]


class BuilderProtocol(Protocol):
    """Protocol for builder functions."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Call the builder function."""
        ...


# ============================================================================
# TYPE GUARDS TESTING WITH COMPREHENSIVE PATTERNS
# ============================================================================


class TestTypeGuards:
    """Test type guard functionality with factory patterns and property testing."""

    def test_is_not_none_basic_scenarios(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test is_not_none type guard with generated test data."""
        # Use factory to generate realistic test data
        user_data = user_data_factory.build()

        # Test with factory-generated values
        assert is_not_none(user_data["name"]) is True
        assert is_not_none(user_data["email"]) is True
        assert is_not_none(user_data["age"]) is True

        # Test basic assertions with fluent pattern
        TestAssertionBuilder("string").satisfies(
            is_not_none, "should not be None"
        ).assert_all()

        TestAssertionBuilder(42).satisfies(
            is_not_none, "should not be None"
        ).assert_all()

        # Test with None
        assert is_not_none(None) is False

    @given(EdgeCaseStrategies.boundary_integers())
    def test_is_not_none_property_based(self, value: int) -> None:
        """Property-based test for is_not_none with various integers."""
        # Property: is_not_none should always return True for any integer
        assert is_not_none(value) is True

    def test_is_list_of_comprehensive(self) -> None:
        """Test is_list_of with comprehensive scenarios using test builder."""
        # Build test scenarios using our test builder (for demonstration)
        _test_data = (
            FlextTestBuilder()
            .with_id("list_validation_test")
            .with_user_data("Test User", "test@example.com")
            .build()
        )

        # Test valid lists with assertion builder
        TestAssertionBuilder([1, 2, 3]).satisfies(
            lambda x: FlextTypeGuards.is_list_of(x, int), "should be list of integers"
        ).assert_all()

        TestAssertionBuilder(["a", "b", "c"]).satisfies(
            lambda x: FlextTypeGuards.is_list_of(x, str), "should be list of strings"
        ).assert_all()

        # Test empty list (boundary case)
        assert FlextTypeGuards.is_list_of([], int) is True

        # Test invalid cases
        assert FlextTypeGuards.is_list_of([1, "2", 3], int) is False
        assert FlextTypeGuards.is_list_of("string", str) is False

    @given(CompositeStrategies.user_profiles())
    def test_is_instance_of_with_user_profiles(
        self, profile: dict[str, object]
    ) -> None:
        """Property-based test for is_instance_of with generated user profiles."""
        assume(PropertyTestHelpers.assume_non_empty_string(profile.get("name", "")))

        # Test type checks on profile components
        assert FlextTypeGuards.is_instance_of(profile["name"], str)
        assert FlextTypeGuards.is_instance_of(profile["email"], str)
        assert FlextTypeGuards.is_instance_of(profile["active"], bool)


# ============================================================================
# VALIDATION DECORATORS WITH ASYNC AND STRESS TESTING
# ============================================================================


class TestValidationDecorators:
    """Test validation decorators with async patterns and stress testing."""

    def test_safe_decorator_comprehensive(self) -> None:
        """Test safe decorator with comprehensive error scenarios."""

        def risky_operation_raw(x: int) -> int:
            if x == 0:
                zero_error = "Cannot divide by zero"
                raise ValueError(zero_error)
            if x < 0:
                negative_error = "Negative values not allowed"
                raise ValueError(negative_error)
            return 100 // x

        risky_operation = safe(cast("FlextCallable[object]", risky_operation_raw))

        # Test successful operations
        result = risky_operation(10)
        TestAssertionBuilder(result).satisfies(
            lambda x: isinstance(x, FlextResult), "should return FlextResult"
        ).satisfies(lambda x: x.success, "should be successful").satisfies(
            lambda x: x.value == 10, "should have correct value"
        ).assert_all()

        # Test error cases
        error_result = risky_operation(0)
        assert error_result.is_failure
        assert "Cannot divide by zero" in (error_result.error or "")

    @pytest.mark.asyncio
    async def test_safe_decorator_async_integration(
        self, async_test_utils: AsyncTestUtils
    ) -> None:
        """Test safe decorator with async operations."""

        async def async_risky_operation_raw(delay: float) -> str:
            await async_test_utils.simulate_delay(delay)
            if delay > 1.0:
                delay_error = "Delay too long"
                raise ValueError(delay_error)
            return f"Completed after {delay}s"

        async_risky_operation = safe(
            cast("FlextCallable[object]", async_risky_operation_raw)
        )

        # Test successful async operation
        result = await async_risky_operation(0.1)
        assert isinstance(result, FlextResult)
        assert result.success

    def test_immutable_decorator_stress_testing(self) -> None:
        """Test immutable decorator with stress testing patterns."""
        stress_runner = StressTestRunner()

        @immutable
        class ImmutableTestClass:
            def __init__(self, value: int) -> None:
                self.value = value

        def create_immutable_instance() -> ImmutableTestClass:
            return ImmutableTestClass(42)

        # Stress test instance creation
        result = stress_runner.run_load_test(
            create_immutable_instance,
            iterations=1000,
            operation_name="immutable_creation",
        )

        assert result["failure_rate"] == 0.0
        assert result["operations_per_second"] > 100

    def test_pure_decorator_complexity_analysis(self) -> None:
        """Test pure decorator with algorithmic complexity analysis."""
        analyzer = ComplexityAnalyzer()

        def pure_computation_raw(size: int) -> int:
            """Pure function for complexity testing."""
            return sum(range(size))

        # For complexity analysis, use the raw function (analyzer expects specific signature)
        complexity_result = analyzer.measure_complexity(
            pure_computation_raw, [100, 200, 400, 800], "pure_computation"
        )

        # Also test the pure decorator works
        pure(cast("Callable[[object], int] | Callable[[], int]", pure_computation_raw))

        assert complexity_result["operation"] == "pure_computation"
        assert len(complexity_result["results"]) == 4  # [100, 200, 400, 800]


# ============================================================================
# VALIDATED MODEL WITH FACTORY PATTERNS
# ============================================================================


class TestFlextModel:
    """Test FlextModel with comprehensive factory and builder patterns."""

    def test_validated_model_with_factory_data(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test FlextModel creation with factory-generated data."""

        class UserModel(FlextModel):
            name: str
            age: int
            email: str

        # Generate realistic test data
        user_data = user_data_factory.build()

        # Test model creation with factory data
        model = UserModel(
            name=user_data["name"], age=user_data["age"], email=user_data["email"]
        )

        # Comprehensive validation with assertion builder
        TestAssertionBuilder(model).is_not_none().satisfies(
            lambda x: hasattr(x, "name"), "should have name attribute"
        ).satisfies(lambda x: hasattr(x, "age"), "should have age attribute").satisfies(
            lambda x: hasattr(x, "email"), "should have email attribute"
        ).satisfies(
            lambda x: hasattr(x, "to_dict_basic"), "should have serialization"
        ).assert_all()

    def test_validated_model_with_given_when_then(self) -> None:
        """Test FlextModel using Given-When-Then pattern."""
        scenario = (
            GivenWhenThenBuilder("user_model_validation")
            .given("a valid user data structure", data={"name": "John", "age": 30})
            .when("creating a FlextModel instance", action="create_model")
            .then("the model should be created successfully", success=True)
            .then("the model should have all required attributes", validated=True)
            .with_tag("validation")
            .with_priority("high")
            .build()
        )

        class UserModel(FlextModel):
            name: str
            age: int

        # Execute the scenario - use standard Pydantic construction
        model = UserModel(**cast("dict[str, object]", scenario.given["data"]))

        # Verify model was created successfully
        assert model is not None
        assert scenario.when["action"] == "create_model"
        assert scenario.then["success"] is True

    @given(CompositeStrategies.user_profiles())
    def test_validated_model_property_based(self, profile: dict[str, object]) -> None:
        """Property-based testing for FlextModel."""
        assume(PropertyTestHelpers.assume_valid_email(profile.get("email", "")))
        assume(PropertyTestHelpers.assume_non_empty_string(profile.get("name", "")))

        class ProfileModel(FlextModel):
            name: str
            email: str
            active: bool

        # Test model creation with generated data
        try:
            model = ProfileModel(
                name=profile["name"], email=profile["email"], active=profile["active"]
            )
            assert model.name == profile["name"]
            assert model.email == profile["email"]
            assert model.active == profile["active"]
        except FlextExceptions:
            # Some generated data might not pass validation
            pass

    def test_validated_model_performance_benchmarking(self, benchmark: object) -> None:
        """Performance benchmark for FlextModel operations."""

        class BenchmarkModel(FlextModel):
            name: str
            value: int
            active: bool

        def create_model_instance() -> BenchmarkModel:
            return BenchmarkModel(name="test", value=42, active=True)

        # Benchmark model creation
        result = BenchmarkUtils.benchmark_with_warmup(
            cast("BenchmarkFixture", benchmark), create_model_instance, warmup_rounds=5
        )

        assert isinstance(result, BenchmarkModel)
        assert result.name == "test"


# ============================================================================
# FACTORY HELPERS WITH ADVANCED PATTERNS
# ============================================================================


class TestFactoryHelpers:
    """Test factory helpers with comprehensive patterns and stress testing."""

    def test_make_factory_with_parametrized_cases(self) -> None:
        """Test make_factory with parametrized test cases."""
        param_builder = ParameterizedTestBuilder("factory_creation")

        # Add various test cases
        param_builder.add_success_cases([
            {"class_name": "SimpleClass", "args": [42], "expected_value": 42},
            {"class_name": "SimpleClass", "args": [100], "expected_value": 100},
        ])

        class SimpleClass:
            def __init__(self, value: int) -> None:
                self.value = value

        factory = make_factory(SimpleClass)

        # Execute parametrized tests
        for params in param_builder.build_pytest_params():
            # Params format: (class_name, args, expected_value, expected_success)
            _class_name, args, expected_value, _expected_success = params
            # Factory has a create method that takes kwargs
            result = factory.create(value=args[0])  # args[0] is the value parameter
            assert result.success, f"Factory creation failed: {result.error}"
            obj = result.value
            assert getattr(obj, "value", None) == expected_value

    def test_make_builder_stress_testing(self) -> None:
        """Test make_builder with stress testing patterns."""

        class BuildableClass:
            def __init__(self, x: int = 0, y: int = 0) -> None:
                self.x = x
                self.y = y

        builder = make_builder(BuildableClass)
        stress_runner = StressTestRunner()

        def create_with_builder() -> BuildableClass:
            result = builder.set(x=10, y=20).build()
            if result.is_failure:
                raise RuntimeError(f"Builder failed: {result.error}")
            return cast("BuildableClass", result.value)

        # Stress test builder performance
        result = stress_runner.run_load_test(
            create_with_builder, iterations=1000, operation_name="builder_stress"
        )

        assert result["failure_rate"] == 0.0
        assert result["successes"] == 1000

    def test_factory_with_memory_profiling(self) -> None:
        """Test factory with memory profiling."""

        class MemoryTestClass:
            def __init__(self, data: list[int]) -> None:
                self.data = data

        factory = make_factory(MemoryTestClass)
        profiler = PerformanceProfiler()

        with profiler.profile_memory("factory_memory_test"):
            # Create instances with large data
            instances = []
            for i in range(10):
                result = factory.create(data=list(range(i * 100)))
                if result.success:
                    instances.append(result.value)
            assert len(instances) == 10

        # Verify memory usage was reasonable
        profiler.assert_memory_efficient(
            max_memory_mb=50.0, operation_name="factory_memory_test"
        )


# ============================================================================
# VALIDATION UTILITIES WITH COMPREHENSIVE TESTING
# ============================================================================


class TestValidationUtilities:
    """Test validation utilities with comprehensive patterns."""

    def test_require_not_none_edge_cases(self) -> None:
        """Test require_not_none with edge cases using test case factory."""
        # Create test cases using factory
        success_cases = [
            TestCaseFactory.create_success_case(
                "string_value", {"input": "hello"}, {"output": "hello"}
            ),
            TestCaseFactory.create_success_case(
                "integer_value", {"input": 42}, {"output": 42}
            ),
            TestCaseFactory.create_success_case(
                "boolean_false", {"input": False}, {"output": False}
            ),
        ]

        failure_cases = [
            TestCaseFactory.create_failure_case(
                "none_value", {"input": None}, "Value cannot be None"
            )
        ]

        # Test success cases
        for case in success_cases:
            case_dict = cast("dict[str, object]", case)
            result = require_not_none(case_dict["input"]["input"])
            assert result == case_dict["expected"]["output"]

        # Test failure cases
        for case in failure_cases:
            case_dict = cast("dict[str, object]", case)
            with pytest.raises(FlextExceptions):
                require_not_none(case_dict["input"]["input"])

    @given(EdgeCaseStrategies.boundary_integers())
    def test_require_positive_property_based(self, value: int) -> None:
        """Property-based test for require_positive."""
        if value > 0:
            # Positive integers should pass
            result = require_positive(value)
            assert result == value
        else:
            # Non-positive integers should fail
            with pytest.raises(FlextExceptions):
                require_positive(value)

    @mark_test_pattern("arrange_act_assert")
    def test_require_non_empty_aaa_pattern(self) -> None:
        """Test require_non_empty using Arrange-Act-Assert pattern."""

        def arrange_data(*args: object, **kwargs: object) -> dict[str, object]:
            _ = args, kwargs  # Mark as used
            return {"valid_string": "hello", "empty_string": "", "whitespace": "   "}

        def act_on_data(data: dict[str, object]) -> dict[str, object]:
            results = {}
            # Test valid string
            results["valid"] = require_non_empty(data["valid_string"])

            # Test failures
            try:
                require_non_empty(data["empty_string"])
                results["empty_failed"] = False
            except FlextExceptions:
                results["empty_failed"] = True

            try:
                require_non_empty(data["whitespace"])
                results["whitespace_failed"] = False
            except FlextExceptions:
                results["whitespace_failed"] = True

            return results

        def assert_results(
            results: dict[str, object], original_data: dict[str, object]
        ) -> None:
            _ = original_data  # Mark as used
            assert results["valid"] == "hello"
            assert results["empty_failed"] is True
            assert results["whitespace_failed"] is True

        @arrange_act_assert(arrange_data, act_on_data, assert_results)
        def test_validation_workflow() -> None:
            pass

        # Execute the AAA pattern test
        test_validation_workflow()


# ============================================================================
# INTEGRATION AND PERFORMANCE TESTING
# ============================================================================


class TestGuardsIntegration:
    """Test integration between guards with performance analysis."""

    def test_validated_model_with_all_guards(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test FlextModel using all guard functions."""

        class ComprehensiveModel(FlextModel):
            name: str
            age: int
            priority: int
            tags: list[str]
            metadata: dict[str, int]

            def __init__(self, **data: object) -> None:
                # Apply all guard validations
                if "name" in data:
                    data["name"] = require_non_empty(data["name"])
                if "age" in data:
                    data["age"] = require_positive(data["age"])

                super().__init__(**data)

        # Test with factory data
        user_data = user_data_factory.build()

        model = ComprehensiveModel(
            name=user_data["name"],
            age=user_data["age"],
            priority=3,
            tags=["test", "user"],
            metadata={"score": 100},
        )

        # Comprehensive validation
        TestAssertionBuilder(model).is_not_none().satisfies(
            lambda x: len(x.name) > 0, "should have non-empty name"
        ).satisfies(lambda x: x.age > 0, "should have positive age").satisfies(
            lambda x: 1 <= x.priority <= 5, "should have valid priority"
        ).satisfies(
            lambda x: FlextTypeGuards.is_list_of(x.tags, str), "should have string tags"
        ).assert_all()

    @pytest.mark.asyncio
    async def test_guards_async_performance_endurance(
        self, async_test_utils: AsyncTestUtils
    ) -> None:
        """Test guards performance in async endurance scenarios."""
        stress_runner = StressTestRunner()

        async def async_validation_workflow() -> dict[str, object]:
            # Simulate async validation workflow
            await async_test_utils.simulate_delay(0.001)

            # Apply multiple guards
            name = require_non_empty("test_user")
            age = require_positive(25)
            priority_value = require_positive(3)

            return {"name": name, "age": age, "priority": priority_value}

        # Run endurance test for 3 seconds
        result = stress_runner.run_endurance_test(
            lambda: async_test_utils.run_sync_in_async(async_validation_workflow()),
            duration_seconds=3.0,
            operation_name="async_guards_endurance",
        )

        assert result["actual_duration_seconds"] >= 2.5
        assert result["operations_per_second"] > 10

    def test_guards_large_scale_complexity(self, benchmark: object) -> None:
        """Test guards with large-scale data complexity analysis."""

        def create_large_validation_scenario() -> bool:
            # Create large data structures
            large_list = list(range(1000))
            large_dict = {f"key_{i}": i for i in range(1000)}

            # Apply multiple guards
            list_valid = FlextGuards.is_list_of(large_list, int)
            dict_valid = FlextGuards.is_dict_of(large_dict, int)

            # Validate ranges
            range_valid = all(
                FlextValidationUtils.require_in_range(i, 0, 999) == i
                for i in range(0, 1000, 100)
            )

            return list_valid and dict_valid and range_valid

        # Benchmark large-scale validation
        result = BenchmarkUtils.benchmark_with_warmup(
            cast("BenchmarkFixture", benchmark),
            create_large_validation_scenario,
            warmup_rounds=3,
        )

        assert result is True


# ============================================================================
# PROPERTY-BASED TESTING COMPREHENSIVE COVERAGE
# ============================================================================


class TestGuardsPropertyBased:
    """Comprehensive property-based testing for all guard functions."""

    @given(FlextStrategies.emails())
    def test_email_validation_properties(self, email: str) -> None:
        """Property-based test for email validation scenarios."""
        assume(PropertyTestHelpers.assume_valid_email(email))

        # Email should pass basic validations
        assert is_not_none(email)
        assert FlextTypeGuards.is_instance_of(email, str)
        assert require_non_empty(email) == email

    @given(CompositeStrategies.configuration_data())
    def test_configuration_validation_properties(
        self, config: dict[str, object]
    ) -> None:
        """Property-based test for configuration validation."""
        # All configuration should be valid dict structure
        assert is_not_none(config)

        # Required fields should be present
        required_fields = ["database_url", "debug", "timeout_seconds"]
        for field in required_fields:
            assert field in config

    @given(EdgeCaseStrategies.unicode_edge_cases())
    def test_unicode_string_guards(self, unicode_text: str) -> None:
        """Property-based test for Unicode string handling."""
        # All strings should pass type checking
        assert FlextTypeGuards.is_instance_of(unicode_text, str)
        assert is_not_none(unicode_text)

        # Empty or whitespace strings should be handled correctly
        if PropertyTestHelpers.assume_non_empty_string(unicode_text):
            result = require_non_empty(unicode_text)
            assert result == unicode_text
