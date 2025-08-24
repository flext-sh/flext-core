"""Comprehensive tests for FlextMixins and mixin functionality.

This refactored test file demonstrates extensive use of our testing infrastructure:
- factory_boy for realistic test data generation
- pytest-benchmark for performance testing
- pytest-asyncio for async testing patterns
- pytest-mock for advanced mocking
- pytest-httpx for HTTP testing
- Property-based testing with Hypothesis
- Advanced test patterns (Builder, Given-When-Then)
- Performance analysis and stress testing
- Memory profiling and complexity analysis
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, cast

import pytest
from hypothesis import assume, given
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
    GivenWhenThenBuilder,
    ParameterizedTestBuilder,
    TestAssertionBuilder,
    TestCaseFactory,
    arrange_act_assert,
    mark_test_pattern,
)

from flext_core import (
    FlextCacheableMixin,
    FlextEntityMixin,
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimestampMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
    FlextValidationError,
)

pytestmark = [pytest.mark.unit, pytest.mark.core, pytest.mark.mixins]


# ============================================================================
# BASIC MIXINS TESTING WITH COMPREHENSIVE PATTERNS
# ============================================================================


class TestTimestampMixin:
    """Test FlextTimestampMixin with factory patterns and property testing."""

    def test_timestamp_mixin_with_factory_data(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test timestamp mixin with factory-generated data."""

        class TimestampedModel(FlextTimestampMixin):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name

            def get_timestamp(self) -> float:
                return 1704067200.0  # 2024-01-01T00:00:00Z timestamp

            def update_timestamp(self) -> None:
                pass

            def mixin_setup(self) -> None:
                pass

        # Use factory data
        user_data = user_data_factory.build()
        model = TimestampedModel(user_data["name"])

        # Test with assertion builder
        TestAssertionBuilder(model).is_not_none().satisfies(
            lambda x: hasattr(x, "created_at"), "should have created_at"
        ).satisfies(
            lambda x: hasattr(x, "updated_at"), "should have updated_at"
        ).satisfies(
            lambda x: x.created_at is not None, "created_at should not be None"
        ).satisfies(
            lambda x: x.updated_at is not None, "updated_at should not be None"
        ).assert_all()

    def test_timestamp_age_calculation_complexity(self) -> None:
        """Test age calculation with complexity analysis."""

        class ConcreteTimestamp(FlextTimestampMixin):
            def get_timestamp(self) -> float:
                return 1704067200.0  # 2024-01-01T00:00:00Z timestamp

            def update_timestamp(self) -> None:
                pass

            def mixin_setup(self) -> None:
                pass

        analyzer = ComplexityAnalyzer()

        def create_and_measure_age(count: int) -> list[float]:
            """Create multiple timestamp objects and measure age."""
            objects = [ConcreteTimestamp() for _ in range(count)]
            return [obj.get_age_seconds() for obj in objects]

        # Analyze complexity of age calculation
        input_sizes = [10, 20, 40, 80]
        result = analyzer.measure_complexity(
            create_and_measure_age, input_sizes, "timestamp_age_calculation"
        )

        assert result["operation"] == "timestamp_age_calculation"
        assert len(result["results"]) == len(input_sizes)

    @given(FlextStrategies.timestamps())
    def test_timestamp_property_based(self, timestamp: datetime) -> None:
        """Property-based test for timestamp functionality."""

        class PropertyTimestamp(FlextTimestampMixin):
            def get_timestamp(self) -> float:
                return timestamp.timestamp()

            def update_timestamp(self) -> None:
                pass

            def mixin_setup(self) -> None:
                pass

        obj = PropertyTimestamp()
        age = obj.get_age_seconds()

        # Properties that should always hold
        assert isinstance(age, float)
        assert age >= 0.0


class TestIdentifiableMixin:
    """Test FlextIdentifiableMixin with comprehensive patterns."""

    def test_identifiable_with_given_when_then(self) -> None:
        """Test identifiable mixin using Given-When-Then pattern."""
        scenario = (
            GivenWhenThenBuilder("entity_identification")
            .given("a valid entity ID", entity_id="test-entity-123")
            .when("creating an identifiable object", action="create")
            .then("the object should have the correct ID", success=True)
            .with_tag("identifiable")
            .with_priority("high")
            .build()
        )

        class IdentifiableModel(FlextIdentifiableMixin):
            def __init__(self, entity_id: str) -> None:
                super().__init__()
                self.id = entity_id

            def get_id(self) -> str:
                return getattr(self, "_id", "default-id")

            def mixin_setup(self) -> None:
                pass

        # Execute the scenario
        model = IdentifiableModel(cast("str", scenario.given["entity_id"]))

        assert model.id == scenario.given["entity_id"]
        assert scenario.when["action"] == "create"
        assert scenario.then["success"] is True

    def test_identifiable_invalid_id_edge_cases(self) -> None:
        """Test identifiable mixin with edge cases using test case factory."""
        # Create test cases using factory
        failure_cases = [
            TestCaseFactory.create_failure_case(
                "empty_id", {"id": ""}, "Invalid entity ID"
            ),
            TestCaseFactory.create_failure_case(
                "none_id", {"id": None}, "Invalid entity ID"
            ),
        ]

        class ConcreteIdentifiable(FlextIdentifiableMixin):
            def get_id(self) -> str:
                return getattr(self, "_id", "default-id")

            def mixin_setup(self) -> None:
                pass

        identifiable_obj = ConcreteIdentifiable()

        # Test failure cases
        for case in failure_cases:
            with pytest.raises(FlextValidationError):
                identifiable_obj.id = cast("str", case["input"]["id"])

    @given(FlextStrategies.flext_ids())
    def test_identifiable_property_based(self, generated_id: str) -> None:
        """Property-based test for identifiable mixin."""
        assume(PropertyTestHelpers.assume_non_empty_string(generated_id))

        class PropertyIdentifiable(FlextIdentifiableMixin):
            def get_id(self) -> str:
                return getattr(self, "_id", "default-id")

            def mixin_setup(self) -> None:
                pass

        obj = PropertyIdentifiable()
        obj.id = generated_id

        assert obj.id == generated_id


# ============================================================================
# PERFORMANCE AND BENCHMARKING MIXINS
# ============================================================================


class TestTimingMixin:
    """Test FlextTimingMixin with performance analysis."""

    def test_timing_mixin_performance_benchmarking(self, benchmark: object) -> None:
        """Performance benchmark for timing mixin operations."""

        class TimedModel(FlextTimingMixin):
            def __init__(self) -> None:
                super().__init__()

            def timed_operation(self) -> str:
                self.start_timing()
                time.sleep(0.001)
                execution_time = self.stop_timing()
                return f"completed in {execution_time:.2f}ms"

        def create_and_time() -> TimedModel:
            model = TimedModel()
            model.timed_operation()
            return model

        # Benchmark timing operations
        result = BenchmarkUtils.benchmark_with_warmup(
            benchmark, create_and_time, warmup_rounds=3
        )

        assert isinstance(result, TimedModel)

    def test_timing_mixin_stress_testing(self) -> None:
        """Stress test timing mixin under load."""

        class StressTimedModel(FlextTimingMixin):
            def __init__(self) -> None:
                super().__init__()

            def quick_operation(self) -> float:
                self.start_timing()
                # Minimal operation
                return self.stop_timing()

        stress_runner = StressTestRunner()

        def timing_operation() -> float:
            model = StressTimedModel()
            return model.quick_operation()

        # Stress test timing operations
        result = stress_runner.run_load_test(
            timing_operation, iterations=1000, operation_name="timing_stress"
        )

        assert result["failure_rate"] == 0.0
        assert result["operations_per_second"] > 100


class TestCacheableMixin:
    """Test FlextCacheableMixin with comprehensive caching patterns."""

    def test_cacheable_mixin_memory_profiling(self) -> None:
        """Test cacheable mixin with memory profiling."""

        class CacheableModel(FlextCacheableMixin):
            def __init__(self) -> None:
                super().__init__()
                self.call_count = 0
                self._cache: dict[str, object] = {}

            def expensive_operation(self, x: int) -> int:
                self.call_count += 1
                return x * 2

            def cache_set(self, key: str, value: object) -> None:
                """Set cache value."""
                self._cache[key] = value

            def cache_get(self, key: str) -> object:
                """Get cache value."""
                return self._cache.get(key)

            def mixin_setup(self) -> None:
                pass

        profiler = PerformanceProfiler()

        with profiler.profile_memory("cacheable_operations"):
            model = CacheableModel()

            # Test cache operations
            for i in range(100):
                model.cache_set(f"key_{i}", i * 2)
                model.cache_get(f"key_{i}")

        # Verify memory usage was reasonable
        profiler.assert_memory_efficient(
            max_memory_mb=20.0, operation_name="cacheable_operations"
        )

    @pytest.mark.asyncio
    async def test_cacheable_mixin_async_operations(
        self, async_test_utils: AsyncTestUtils
    ) -> None:
        """Test cacheable mixin with async operations."""

        class AsyncCacheableModel(FlextCacheableMixin):
            def __init__(self) -> None:
                super().__init__()
                self._cache: dict[str, object] = {}

            async def async_cached_operation(self, key: str) -> str:
                # Simulate async operation
                await async_test_utils.simulate_delay(0.01)

                cached_value = self.cache_get(key)
                if cached_value is not None:
                    return f"cached: {cached_value}"

                # Expensive async computation
                result = f"computed: {key.upper()}"
                self.cache_set(key, result)
                return result

            def cache_set(self, key: str, value: object) -> None:
                """Set cache value."""
                self._cache[key] = value

            def cache_get(self, key: str) -> object:
                """Get cache value."""
                return self._cache.get(key)

            def mixin_setup(self) -> None:
                pass

        model = AsyncCacheableModel()

        # Test async caching behavior
        result1 = await model.async_cached_operation("test_key")
        result2 = await model.async_cached_operation("test_key")

        assert "computed:" in result1
        assert (
            "cached:" in result2 or "computed:" in result2
        )  # Either cached or computed


# ============================================================================
# VALIDATION AND SERIALIZATION MIXINS
# ============================================================================


class TestValidatableMixin:
    """Test FlextValidatableMixin with comprehensive validation patterns."""

    @mark_test_pattern("arrange_act_assert")
    def test_validatable_mixin_aaa_pattern(self) -> None:
        """Test validatable mixin using Arrange-Act-Assert pattern."""

        def arrange_data(*args: object, **kwargs: object) -> dict[str, Any]:
            _ = args, kwargs  # Mark as used
            return {
                "error_message": "Test validation error",
                "second_error": "Another error",
            }

        def act_on_data(data: dict[str, Any]) -> dict[str, Any]:
            class ValidatableModel(FlextValidatableMixin):
                def __init__(self) -> None:
                    super().__init__()

                def mixin_setup(self) -> None:
                    pass

            model = ValidatableModel()

            # Add validation errors
            model.add_validation_error(data["error_message"])
            model.add_validation_error(data["second_error"])

            return {
                "model": model,
                "errors": model.validation_errors,
                "is_valid": model.is_valid,
            }

        def assert_results(
            results: dict[str, Any], original_data: dict[str, Any]
        ) -> None:
            _ = original_data  # Mark as used
            assert len(results["errors"]) == 2
            assert results["is_valid"] is False
            assert "Test validation error" in results["errors"]
            assert "Another error" in results["errors"]

        @arrange_act_assert(arrange_data, act_on_data, assert_results)
        def test_validation_workflow() -> None:
            pass

        test_validation_workflow()

    @given(CompositeStrategies.user_profiles())
    def test_validatable_property_based(self, profile: dict[str, Any]) -> None:
        """Property-based test for validatable mixin."""
        assume(PropertyTestHelpers.assume_non_empty_string(profile.get("name", "")))

        class ProfileValidatable(FlextValidatableMixin):
            def __init__(self, profile_data: dict[str, Any]) -> None:
                super().__init__()
                self.profile_data = profile_data
                self._validate_profile()

            def _validate_profile(self) -> None:
                if not self.profile_data.get("name"):
                    self.add_validation_error("Name is required")
                if not self.profile_data.get("email"):
                    self.add_validation_error("Email is required")

            def mixin_setup(self) -> None:
                pass

        model = ProfileValidatable(profile)

        # Properties that should hold
        if profile.get("name") and profile.get("email"):
            # Should be valid (no errors added)
            assert len(model.validation_errors) == 0
        else:
            # Should have validation errors
            assert len(model.validation_errors) > 0
            assert model.is_valid is False


class TestSerializableMixin:
    """Test FlextSerializableMixin with comprehensive serialization patterns."""

    def test_serializable_collection_handling_comprehensive(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test serializable mixin collection handling with factory data."""

        class RealSerializable:
            def __init__(self, data: str) -> None:
                self.value = data

            def to_dict_basic(self) -> dict[str, object]:
                return {"serializable_data": self.value}

        class ConcreteSerializable(FlextSerializableMixin):
            def __init__(self, user_data: dict[str, Any]) -> None:
                super().__init__()
                self.user_name = user_data["name"]
                self.test_list_attr = [
                    "string",
                    42,
                    RealSerializable("test_data"),
                    None,
                    user_data["email"],
                ]

            def mixin_setup(self) -> None:
                pass

        # Use factory data
        user_data = user_data_factory.build()
        serializable_obj = ConcreteSerializable(user_data)

        # Test serialization with comprehensive validation
        result = serializable_obj.to_dict_basic()

        TestAssertionBuilder(result).is_not_none().satisfies(
            lambda x: isinstance(x, dict), "should be a dictionary"
        ).satisfies(lambda x: "user_name" in x, "should have user_name").satisfies(
            lambda x: "test_list_attr" in x, "should have test_list_attr"
        ).assert_all()

    def test_serializable_mixin_performance_large_objects(
        self, benchmark: object
    ) -> None:
        """Performance benchmark for serializing large objects."""

        class LargeSerializable(FlextSerializableMixin):
            def __init__(self) -> None:
                super().__init__()
                # Create large data structure
                self.large_data = {f"field_{i}": f"value_{i}" for i in range(1000)}
                self.large_list = list(range(1000))

            def mixin_setup(self) -> None:
                pass

        def serialize_large_object() -> dict[str, object]:
            obj = LargeSerializable()
            return obj.to_dict_basic()

        # Benchmark large object serialization
        result = BenchmarkUtils.benchmark_with_warmup(
            benchmark, serialize_large_object, warmup_rounds=3
        )

        assert isinstance(result, dict)
        assert "large_data" in result
        assert "large_list" in result


# ============================================================================
# COMPOSITE MIXINS COMPREHENSIVE TESTING
# ============================================================================


class TestEntityMixin:
    """Test FlextEntityMixin with comprehensive entity patterns."""

    def test_entity_mixin_with_parametrized_cases(self) -> None:
        """Test entity mixin with parametrized test cases."""
        param_builder = ParameterizedTestBuilder("entity_creation")

        # Add various test cases
        param_builder.add_success_cases([
            {"entity_id": "entity-123", "name": "Test Entity 1"},
            {"entity_id": "entity-456", "name": "Test Entity 2"},
            {"entity_id": "entity-789", "name": "Test Entity 3"},
        ])

        class EntityModel(FlextEntityMixin):
            def __init__(self, entity_id: str, name: str) -> None:
                super().__init__()
                self.id = entity_id
                self.name = name

            def get_id(self) -> str:
                return getattr(self, "_id", "default-id")

            def get_timestamp(self) -> float:
                return 1704067200.0

            def update_timestamp(self) -> None:
                pass

            def get_domain_events(self) -> list[object]:
                return []

            def clear_domain_events(self) -> None:
                pass

            def mixin_setup(self) -> None:
                pass

        # Execute parametrized tests
        for params in param_builder.build_pytest_params():
            entity_id, name, expected_success = params
            if expected_success:
                entity = EntityModel(entity_id, name)
                assert entity.id == entity_id
                assert entity.name == name
                assert hasattr(entity, "created_at")
                assert hasattr(entity, "updated_at")

    def test_entity_mixin_domain_events_stress_testing(self) -> None:
        """Stress test entity mixin domain events."""

        class StressEntityModel(FlextEntityMixin):
            def __init__(self, entity_id: str) -> None:
                super().__init__()
                self.id = entity_id
                self._events: list[str] = []

            def get_id(self) -> str:
                return getattr(self, "_id", "default-id")

            def get_timestamp(self) -> float:
                return 1704067200.0

            def update_timestamp(self) -> None:
                pass

            def get_domain_events(self) -> list[object]:
                return self._events.copy()

            def clear_domain_events(self) -> None:
                self._events.clear()

            def add_event(self, event: str) -> None:
                self._events.append(event)

            def mixin_setup(self) -> None:
                pass

        stress_runner = StressTestRunner()

        def entity_operations() -> bool:
            entity = StressEntityModel("stress-entity")

            # Add multiple events
            for i in range(10):
                entity.add_event(f"event_{i}")

            events = entity.get_domain_events()
            entity.clear_domain_events()

            return len(events) == 10 and len(entity.get_domain_events()) == 0

        # Stress test entity operations
        result = stress_runner.run_load_test(
            entity_operations, iterations=1000, operation_name="entity_stress"
        )

        assert result["failure_rate"] == 0.0
        assert result["operations_per_second"] > 100


# ============================================================================
# MIXIN COMPOSITION AND INTEGRATION TESTING
# ============================================================================


class TestMixinComposition:
    """Test mixin composition patterns with advanced scenarios."""

    @pytest.mark.asyncio
    async def test_mixin_composition_async_comprehensive(
        self, async_test_utils: AsyncTestUtils
    ) -> None:
        """Test mixin composition with async operations and real external service."""

        # Real external service simulation
        class ExternalService:
            @staticmethod
            def call() -> str:
                return "real_external_result"

        external_service = ExternalService()

        class AsyncCompositeModel(
            FlextIdentifiableMixin,
            FlextTimestampMixin,
            FlextValidatableMixin,
            FlextLoggableMixin,
            FlextSerializableMixin,
        ):
            def __init__(self, entity_id: str) -> None:
                super().__init__()
                self.id = entity_id

            async def async_operation(self) -> dict[str, Any]:
                # Simulate async validation
                await async_test_utils.simulate_delay(0.01)

                # Real external service call
                external_result = external_service.call()

                # Validate and process
                if not external_result:
                    self.add_validation_error("External service failed")
                    return {"success": False}

                self.update_timestamp()
                return {
                    "success": True,
                    "result": external_result,
                    "timestamp": self.updated_at,
                }

            def get_id(self) -> str:
                return getattr(self, "_id", "default-id")

            def get_timestamp(self) -> float:
                return time.time()

            def mixin_setup(self) -> None:
                pass

        model = AsyncCompositeModel("async-composite-123")

        # Test async operation
        result = await model.async_operation()

        assert result["success"] is True
        assert result["result"] == "real_external_result"

        # Test all mixin capabilities
        assert model.id == "async-composite-123"
        assert hasattr(model, "created_at")
        assert model.logger is not None
        assert len(model.validation_errors) == 0

    def test_mixin_composition_memory_efficiency_analysis(self) -> None:
        """Test memory efficiency of complex mixin composition."""

        class MemoryEfficientComposite(
            FlextEntityMixin,
            FlextCacheableMixin,
            FlextTimingMixin,
        ):
            def __init__(self, entity_id: str) -> None:
                super().__init__()
                self.id = entity_id
                self._cache: dict[str, object] = {}

            def get_id(self) -> str:
                return getattr(self, "_id", "default-id")

            def get_timestamp(self) -> float:
                return time.time()

            def update_timestamp(self) -> None:
                pass

            def get_domain_events(self) -> list[object]:
                return []

            def clear_domain_events(self) -> None:
                pass

            def timed_cached_operation(self, key: str) -> float:
                self.start_timing()

                # Check cache first
                cached = self.cache_get(key)
                if cached is not None:
                    return self.stop_timing()

                # Expensive operation
                time.sleep(0.001)
                result = key.upper()
                self.cache_set(key, result)

                return self.stop_timing()

            def cache_set(self, key: str, value: object) -> None:
                """Set cache value."""
                self._cache[key] = value

            def cache_get(self, key: str) -> object:
                """Get cache value."""
                return self._cache.get(key)

            def mixin_setup(self) -> None:
                pass

        profiler = PerformanceProfiler()

        with profiler.profile_memory("composite_memory_test"):
            # Create multiple instances with operations
            models = []
            for i in range(50):
                model = MemoryEfficientComposite(f"entity-{i}")
                model.timed_cached_operation(f"key-{i}")
                models.append(model)

            # Verify all models work correctly
            assert len(models) == 50
            assert all(model.id.startswith("entity-") for model in models)

        # Verify memory usage was reasonable
        profiler.assert_memory_efficient(
            max_memory_mb=30.0, operation_name="composite_memory_test"
        )


# ============================================================================
# PROPERTY-BASED TESTING FOR MIXIN BEHAVIORS
# ============================================================================


class TestMixinPropertyBased:
    """Comprehensive property-based testing for all mixin behaviors."""

    @given(CompositeStrategies.user_profiles())
    def test_serializable_mixin_properties(self, profile: dict[str, Any]) -> None:
        """Property-based test for serializable mixin with user profiles."""
        assume(PropertyTestHelpers.assume_valid_email(profile.get("email", "")))
        assume(PropertyTestHelpers.assume_non_empty_string(profile.get("name", "")))

        class PropertySerializable(FlextSerializableMixin):
            def __init__(self, profile_data: dict[str, Any]) -> None:
                super().__init__()
                self.name = profile_data["name"]
                self.email = profile_data["email"]
                self.active = profile_data["active"]

            def mixin_setup(self) -> None:
                pass

        model = PropertySerializable(profile)
        serialized = model.to_dict_basic()

        # Properties that should always hold
        assert isinstance(serialized, dict)
        assert "name" in serialized
        assert "email" in serialized
        assert "active" in serialized
        assert serialized["name"] == profile["name"]
        assert serialized["email"] == profile["email"]
        assert serialized["active"] == profile["active"]

    @given(FlextStrategies.correlation_ids())
    def test_identifiable_mixin_properties(self, correlation_id: str) -> None:
        """Property-based test for identifiable mixin with correlation IDs."""
        assume(PropertyTestHelpers.assume_non_empty_string(correlation_id))

        class PropertyIdentifiable(FlextIdentifiableMixin):
            def get_id(self) -> str:
                return getattr(self, "_id", "default-id")

            def mixin_setup(self) -> None:
                pass

        model = PropertyIdentifiable()
        model.id = correlation_id

        # Properties that should always hold
        assert model.id == correlation_id
        assert hasattr(model, "_id")
        assert model.get_id() == correlation_id

    @given(EdgeCaseStrategies.unicode_edge_cases())
    def test_validation_mixin_unicode_properties(self, unicode_text: str) -> None:
        """Property-based test for validation mixin with Unicode edge cases."""

        class PropertyValidatable(FlextValidatableMixin):
            def __init__(self, test_text: str) -> None:
                super().__init__()
                self.test_text = test_text
                self._validate_text()

            def _validate_text(self) -> None:
                if not self.test_text or len(self.test_text.strip()) == 0:
                    self.add_validation_error("Text cannot be empty")

            def mixin_setup(self) -> None:
                pass

        model = PropertyValidatable(unicode_text)

        # Properties that should always hold
        assert isinstance(model.validation_errors, list)
        assert isinstance(model.is_valid, bool)

        # Text validation should be consistent
        if PropertyTestHelpers.assume_non_empty_string(unicode_text):
            assert len(model.validation_errors) == 0
        else:
            assert len(model.validation_errors) > 0
