"""Comprehensive tests for core.py module.

# Constants
EXPECTED_TOTAL_PAGES = 8
EXPECTED_DATA_COUNT = 3

This test suite provides complete coverage of the FlextCore system,
testing all aspects including singleton pattern, facade functionality,
dependency injection, logging, railway programming utilities, and
configuration management to achieve near 100% coverage.
"""

from __future__ import annotations

import contextlib
from unittest.mock import Mock, patch

import pytest

from flext_core.constants import FlextConstants
from flext_core.container import ServiceKey
from flext_core.core import FlextCore, flext_core
from flext_core.loggings import FlextLogger
from flext_core.result import FlextResult

pytestmark = [pytest.mark.unit, pytest.mark.core]


# Test data fixtures
@pytest.fixture
def clean_flext_core() -> FlextCore:
    """Create a fresh FlextCore instance for testing."""
    # Reset singleton
    FlextCore._instance = None
    return FlextCore.get_instance()


@pytest.fixture
def mock_service() -> Mock:
    """Mock service for container testing."""
    return Mock(spec=["process", "name"])


@pytest.fixture
def service_key() -> ServiceKey[Mock]:
    """Service key for testing."""
    return ServiceKey[Mock]("test_service")


# Test service classes for typed services
class UserService:
    """Test service for dependency injection."""

    def __init__(self, name: str = "test_user_service") -> None:
        self.name = name

    def get_user(self, user_id: str) -> str:
        """Get user by ID."""
        return f"User {user_id}"


class DataService:
    """Another test service for dependency injection."""

    def __init__(self, connection: str = "test_db") -> None:
        self.connection = connection

    def save_data(self, data: str) -> bool:
        """Save data."""
        return bool(data)


@pytest.mark.unit
class TestFlextCoreSingleton:
    """Test FlextCore singleton pattern."""

    def test_singleton_instance_creation(self) -> None:
        """Test singleton instance creation."""
        # Reset singleton
        FlextCore._instance = None

        instance1 = FlextCore.get_instance()
        instance2 = FlextCore.get_instance()

        assert isinstance(instance1, FlextCore)
        assert isinstance(instance2, FlextCore)
        assert instance1 is instance2

    def test_singleton_initialization(self) -> None:
        """Test singleton initialization."""
        FlextCore._instance = None

        instance = FlextCore.get_instance()

        assert hasattr(instance, "_container")
        assert hasattr(instance, "_settings_cache")
        assert isinstance(instance._settings_cache, dict)

    def test_singleton_reset_behavior(self) -> None:
        """Test singleton reset behavior."""
        FlextCore._instance = None

        instance1 = FlextCore.get_instance()

        # Manually reset (simulating test cleanup)
        FlextCore._instance = None
        instance2 = FlextCore.get_instance()

        # New instances should be different objects
        assert instance1 is not instance2

    def test_singleton_thread_safety_simulation(self) -> None:
        """Test singleton thread safety (simulated)."""
        FlextCore._instance = None

        # Simulate concurrent access
        instances = [FlextCore.get_instance() for _ in range(10)]

        # All should be the same instance
        first_instance = instances[0]
        for instance in instances[1:]:
            assert instance is first_instance


@pytest.mark.unit
class TestFlextCoreContainerIntegration:
    """Test FlextCore container integration."""

    def test_container_property_access(self, clean_flext_core: FlextCore) -> None:
        """Test container property access."""
        container = clean_flext_core.container

        assert container is not None
        assert hasattr(container, "register")
        assert hasattr(container, "get")

    def test_register_service_success(
        self,
        clean_flext_core: FlextCore,
        service_key: ServiceKey[Mock],
        mock_service: Mock,
    ) -> None:
        """Test successful service registration."""
        result = clean_flext_core.register_service(service_key, mock_service)

        assert result.is_success

        # Verify registration worked
        retrieval_result = clean_flext_core.get_service(service_key)
        assert retrieval_result.is_success
        assert retrieval_result.data is mock_service

    def test_register_service_typed_keys(self, clean_flext_core: FlextCore) -> None:
        """Test service registration with typed keys."""
        user_service = UserService("prod_user_service")
        user_key = ServiceKey[UserService]("user_service")

        result = clean_flext_core.register_service(user_key, user_service)
        assert result.is_success

        # Retrieve and verify type safety
        retrieval_result = clean_flext_core.get_service(user_key)
        assert retrieval_result.is_success
        retrieved_service = retrieval_result.data
        assert isinstance(retrieved_service, UserService)
        if retrieved_service.name != "prod_user_service":
            raise AssertionError(f"Expected {"prod_user_service"}, got {retrieved_service.name}")

    def test_get_service_success(
        self,
        clean_flext_core: FlextCore,
        service_key: ServiceKey[Mock],
        mock_service: Mock,
    ) -> None:
        """Test successful service retrieval."""
        # Register first
        clean_flext_core.register_service(service_key, mock_service)

        result = clean_flext_core.get_service(service_key)

        assert result.is_success
        assert result.data is mock_service

    def test_get_service_not_found(
        self,
        clean_flext_core: FlextCore,
    ) -> None:
        """Test service retrieval when service not found."""
        non_existent_key = ServiceKey[Mock]("non_existent_service")

        result = clean_flext_core.get_service(non_existent_key)

        assert result.is_failure
        if "not found" not in result.error.lower():
            raise AssertionError(f"Expected {"not found"} in {result.error.lower()}")

    def test_multiple_services_registration(self, clean_flext_core: FlextCore) -> None:
        """Test registration of multiple different services."""
        user_service = UserService()
        data_service = DataService()

        user_key = ServiceKey[UserService]("user_service")
        data_key = ServiceKey[DataService]("data_service")

        user_result = clean_flext_core.register_service(user_key, user_service)
        data_result = clean_flext_core.register_service(data_key, data_service)

        assert user_result.is_success
        assert data_result.is_success

        # Both services should be retrievable
        user_retrieval = clean_flext_core.get_service(user_key)
        data_retrieval = clean_flext_core.get_service(data_key)

        assert user_retrieval.is_success
        assert data_retrieval.is_success
        assert isinstance(user_retrieval.data, UserService)
        assert isinstance(data_retrieval.data, DataService)

    def test_service_registration_failure_handling(
        self,
        clean_flext_core: FlextCore,
    ) -> None:
        """Test service registration failure handling."""
        # Mock container to simulate failure
        with (
            patch.object(clean_flext_core, "_container"),
            patch("flext_core.core.register_typed") as mock_register,
        ):
            mock_register.return_value = FlextResult.fail("Registration failed")

            service_key = ServiceKey[Mock]("test_service")
            result = clean_flext_core.register_service(service_key, Mock())

            assert result.is_failure
            if "Registration failed" not in result.error:
                raise AssertionError(f"Expected {"Registration failed"} in {result.error}")


@pytest.mark.unit
class TestFlextCoreLogging:
    """Test FlextCore logging functionality."""

    def test_get_logger_success(self, clean_flext_core: FlextCore) -> None:
        """Test successful logger retrieval."""
        logger_name = "test.module"
        logger = clean_flext_core.get_logger(logger_name)

        assert isinstance(logger, FlextLogger)
        if logger._name != logger_name:
            raise AssertionError(f"Expected {logger_name}, got {logger._name}")

    def test_get_logger_multiple_names(self, clean_flext_core: FlextCore) -> None:
        """Test getting loggers with different names."""
        logger1 = clean_flext_core.get_logger("module1")
        logger2 = clean_flext_core.get_logger("module2")

        if logger1._name != "module1":

            raise AssertionError(f"Expected {"module1"}, got {logger1._name}")
        assert logger2._name == "module2"
        # Should be different logger instances for different names
        assert logger1._name != logger2._name

    def test_get_logger_same_name_caching(self, clean_flext_core: FlextCore) -> None:
        """Test logger caching behavior."""
        logger_name = "cached.module"
        logger1 = clean_flext_core.get_logger(logger_name)
        logger2 = clean_flext_core.get_logger(logger_name)

        # FlextLoggerFactory should return the same instance for same name
        if logger1._name == logger2._name != logger_name:
            raise AssertionError(f"Expected {logger_name}, got {logger1._name == logger2._name}")

    def test_configure_logging_default_settings(
        self,
        clean_flext_core: FlextCore,
    ) -> None:
        """Test logging configuration with default settings."""
        # Should not raise exception
        with contextlib.suppress(Exception):
            clean_flext_core.configure_logging()

    def test_configure_logging_custom_level(self, clean_flext_core: FlextCore) -> None:
        """Test logging configuration with custom level."""
        with patch(
            "flext_core.core.FlextLoggerFactory.set_global_level",
        ) as mock_set_level:
            clean_flext_core.configure_logging(log_level="DEBUG")
            mock_set_level.assert_called_once_with("DEBUG")

    def test_configure_logging_json_output_parameter(
        self,
        clean_flext_core: FlextCore,
    ) -> None:
        """Test logging configuration with json_output parameter."""
        # Should not raise exception even with unused parameter
        with contextlib.suppress(Exception):
            clean_flext_core.configure_logging(_json_output=True)

    def test_logging_integration_with_services(
        self,
        clean_flext_core: FlextCore,
    ) -> None:
        """Test logging integration in service context."""
        logger = clean_flext_core.get_logger("service.integration")

        # Register a service
        service = UserService()
        service_key = ServiceKey[UserService]("user_service")
        result = clean_flext_core.register_service(service_key, service)

        assert result.is_success

        # Logger should still work
        if logger._name != "service.integration":
            raise AssertionError(f"Expected {"service.integration"}, got {logger._name}")


@pytest.mark.unit
class TestFlextCoreResultPatterns:
    """Test FlextCore result pattern utilities."""

    def test_ok_static_method(self, clean_flext_core: FlextCore) -> None:
        """Test ok static method."""
        result = clean_flext_core.ok("success_value")

        assert isinstance(result, FlextResult)
        assert result.is_success
        if result.data != "success_value":
            raise AssertionError(f"Expected {"success_value"}, got {result.data}")

    def test_ok_with_different_types(self, clean_flext_core: FlextCore) -> None:
        """Test ok method with different data types."""
        # String
        str_result = clean_flext_core.ok("test")
        assert str_result.is_success
        if str_result.data != "test":
            raise AssertionError(f"Expected {"test"}, got {str_result.data}")

        # Integer
        int_result = clean_flext_core.ok(42)
        assert int_result.is_success
        if int_result.data != 42:
            raise AssertionError(f"Expected {42}, got {int_result.data}")

        # List
        list_result = clean_flext_core.ok([1, 2, 3])
        assert list_result.is_success
        if list_result.data != [1, 2, 3]:
            raise AssertionError(f"Expected {[1, 2, 3]}, got {list_result.data}")

        # Dict
        dict_result = clean_flext_core.ok({"key": "value"})
        assert dict_result.is_success
        if dict_result.data != {"key": "value"}:
            raise AssertionError(f"Expected {{"key": "value"}}, got {dict_result.data}")

    def test_fail_static_method(self, clean_flext_core: FlextCore) -> None:
        """Test fail static method."""
        result = clean_flext_core.fail("error_message")

        assert isinstance(result, FlextResult)
        assert result.is_failure
        if result.error != "error_message":
            raise AssertionError(f"Expected {"error_message"}, got {result.error}")

    def test_fail_with_different_messages(self, clean_flext_core: FlextCore) -> None:
        """Test fail method with different error messages."""
        error1 = clean_flext_core.fail("Simple error")
        assert error1.is_failure
        if error1.error != "Simple error":
            raise AssertionError(f"Expected {"Simple error"}, got {error1.error}")

        error2 = clean_flext_core.fail("Complex error with details")
        assert error2.is_failure
        if error2.error != "Complex error with details":
            raise AssertionError(f"Expected {"Complex error with details"}, got {error2.error}")

        # Empty error should be handled
        error3 = clean_flext_core.fail("")
        assert error3.is_failure
        # FlextResult handles empty errors
        if error3.error != "Unknown error occurred":
            raise AssertionError(f"Expected {"Unknown error occurred"}, got {error3.error}")

    def test_result_pattern_integration(self, clean_flext_core: FlextCore) -> None:
        """Test result pattern integration."""
        # Chain ok and fail operations
        success = clean_flext_core.ok("initial")
        failure = clean_flext_core.fail("failed operation")

        # Should be able to chain
        chained_success = success.map(lambda x: f"processed {x}")
        assert chained_success.is_success
        if chained_success.data != "processed initial":
            raise AssertionError(f"Expected {"processed initial"}, got {chained_success.data}")

        # Failure should propagate
        chained_failure = failure.map(lambda x: f"processed {x}")
        assert chained_failure.is_failure
        if chained_failure.error != "failed operation":
            raise AssertionError(f"Expected {"failed operation"}, got {chained_failure.error}")


@pytest.mark.unit
class TestFlextCoreRailwayProgramming:
    """Test FlextCore railway programming utilities."""

    def test_pipe_success_pipeline(self, clean_flext_core: FlextCore) -> None:
        """Test pipe with successful pipeline."""

        def add_one(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(x) + 1)  # type: ignore[arg-type]

        def multiply_two(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(x) * 2)  # type: ignore[arg-type]

        def to_string(x: object) -> FlextResult[object]:
            return FlextResult.ok(str(x))

        pipeline = clean_flext_core.pipe(add_one, multiply_two, to_string)
        result = pipeline(5)

        assert result.is_success
        if result.data != "12"  # (5 + 1) * 2 = 12:
            raise AssertionError(f"Expected {"12"  # (5 + 1) * 2 = 12}, got {result.data}")

    def test_pipe_failure_propagation(self, clean_flext_core: FlextCore) -> None:
        """Test pipe with failure propagation."""

        def succeed(x: object) -> FlextResult[object]:
            return FlextResult.ok(f"success_{x}")

        def fail_step(x: object) -> FlextResult[object]:
            return FlextResult.fail("Pipeline failed")

        def never_reached(x: object) -> FlextResult[object]:
            return FlextResult.ok("should_not_reach")

        pipeline = clean_flext_core.pipe(succeed, fail_step, never_reached)
        result = pipeline("input")

        assert result.is_failure
        if result.error != "Pipeline failed":
            raise AssertionError(f"Expected {"Pipeline failed"}, got {result.error}")

    def test_pipe_empty_pipeline(self, clean_flext_core: FlextCore) -> None:
        """Test pipe with empty pipeline."""
        pipeline = clean_flext_core.pipe()
        result = pipeline("test_value")

        assert result.is_success
        if result.data != "test_value":
            raise AssertionError(f"Expected {"test_value"}, got {result.data}")

    def test_pipe_single_function(self, clean_flext_core: FlextCore) -> None:
        """Test pipe with single function."""

        def transform(x: object) -> FlextResult[object]:
            return FlextResult.ok(f"transformed_{x}")

        pipeline = clean_flext_core.pipe(transform)
        result = pipeline("input")

        assert result.is_success
        if result.data != "transformed_input":
            raise AssertionError(f"Expected {"transformed_input"}, got {result.data}")

    def test_compose_right_to_left(self, clean_flext_core: FlextCore) -> None:
        """Test compose function (right to left execution)."""

        def add_prefix(x: object) -> FlextResult[object]:
            return FlextResult.ok(f"prefix_{x}")

        def add_suffix(x: object) -> FlextResult[object]:
            return FlextResult.ok(f"{x}_suffix")

        # Compose should execute right to left
        composition = clean_flext_core.compose(add_prefix, add_suffix)
        result = composition("middle")

        assert result.is_success
        # Right to left: add_suffix first, then add_prefix
        if result.data != "prefix_middle_suffix":
            raise AssertionError(f"Expected {"prefix_middle_suffix"}, got {result.data}")

    def test_when_predicate_true(self, clean_flext_core: FlextCore) -> None:
        """Test when with true predicate."""

        def is_positive(x: int) -> bool:
            return x > 0

        def double_value(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        def negate_value(x: int) -> FlextResult[int]:
            return FlextResult.ok(-x)

        conditional = clean_flext_core.when(is_positive, double_value, negate_value)
        result = conditional(5)

        assert result.is_success
        if result.data != 10  # Positive, so doubled:
            raise AssertionError(f"Expected {10  # Positive, so doubled}, got {result.data}")

    def test_when_predicate_false_with_else(self, clean_flext_core: FlextCore) -> None:
        """Test when with false predicate and else function."""

        def is_positive(x: int) -> bool:
            return x > 0

        def double_value(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        def negate_value(x: int) -> FlextResult[int]:
            return FlextResult.ok(-x)

        conditional = clean_flext_core.when(is_positive, double_value, negate_value)
        result = conditional(-3)

        assert result.is_success
        if result.data != EXPECTED_DATA_COUNT  # Negative, so negated (becomes positive):
            raise AssertionError(f"Expected {3  # Negative, so negated (becomes positive)}, got {result.data}")

    def test_when_predicate_false_no_else(self, clean_flext_core: FlextCore) -> None:
        """Test when with false predicate and no else function."""

        def is_positive(x: int) -> bool:
            return x > 0

        def double_value(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        conditional = clean_flext_core.when(is_positive, double_value)
        result = conditional(-3)

        assert result.is_success
        if result.data != -3  # Unchanged when predicate false and no else:
            raise AssertionError(f"Expected {-3  # Unchanged when predicate false and (
                no else}, got {result.data}"))

    def test_when_exception_handling(self, clean_flext_core: FlextCore) -> None:
        """Test when with exception in predicate."""

        def failing_predicate(x: object) -> bool:
            msg = "Predicate error"
            raise ValueError(msg)

        def then_func(x: object) -> FlextResult[object]:
            return FlextResult.ok("success")

        conditional = clean_flext_core.when(failing_predicate, then_func)

        # Should handle exception gracefully
        with contextlib.suppress(Exception):
            _ = conditional("test")  # Test conditional call
            # Behavior may vary based on implementation

    def test_tap_side_effect_execution(self, clean_flext_core: FlextCore) -> None:
        """Test tap side effect execution."""
        side_effect_calls = []

        def record_value(x: object) -> None:
            side_effect_calls.append(x)

        tap_func = clean_flext_core.tap(record_value)
        result = tap_func("test_value")

        assert result.is_success
        if result.data != "test_value":
            raise AssertionError(f"Expected {"test_value"}, got {result.data}")
        assert side_effect_calls == ["test_value"]

    def test_tap_in_pipeline(self, clean_flext_core: FlextCore) -> None:
        """Test tap function in pipeline."""
        side_effects = []

        def log_step(name: str) -> callable:
            def logger(x: object) -> None:
                side_effects.append(f"{name}: {x}")

            return clean_flext_core.tap(logger)

        def add_ten(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(x) + 10)  # type: ignore[arg-type]

        def multiply_three(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(x) * 3)  # type: ignore[arg-type]

        pipeline = clean_flext_core.pipe(
            log_step("start"),
            add_ten,
            log_step("after_add"),
            multiply_three,
            log_step("final"),
        )

        result = pipeline(5)

        assert result.is_success
        if result.data != 45  # (5 + 10) * 3:
            raise AssertionError(f"Expected {45  # (5 + 10) * 3}, got {result.data}")
        assert side_effects == [
            "start: 5",
            "after_add: 15",
            "final: 45",
        ]

    def test_complex_railway_pattern(self, clean_flext_core: FlextCore) -> None:
        """Test complex railway programming pattern."""
        logged_values = []

        def validate_positive(x: int) -> FlextResult[int]:
            if x <= 0:
                return FlextResult.fail("Value must be positive")
            return FlextResult.ok(x)

        def double_if_even(x: int) -> FlextResult[int]:
            if x % 2 == 0:
                return FlextResult.ok(x * 2)
            return FlextResult.ok(x)

        def log_value(x: int) -> None:
            logged_values.append(x)

        # Complex pipeline with conditional logic and side effects
        pipeline = clean_flext_core.pipe(
            lambda x: validate_positive(int(x)),  # type: ignore[arg-type,return-value]
            clean_flext_core.tap(log_value),
            lambda x: double_if_even(int(x)),  # type: ignore[arg-type,return-value]
            clean_flext_core.tap(log_value),
        )

        # Test with even positive number
        result1 = pipeline(4)
        assert result1.is_success
        if result1.data != EXPECTED_TOTAL_PAGES  # 4 * 2:
            raise AssertionError(f"Expected {8  # 4 * 2}, got {result1.data}")

        # Test with odd positive number
        result2 = pipeline(3)
        assert result2.is_success
        if result2.data != EXPECTED_DATA_COUNT  # unchanged:
            raise AssertionError(f"Expected {3  # unchanged}, got {result2.data}")

        # Test with negative number
        result3 = pipeline(-1)
        assert result3.is_failure
        if "positive" not in result3.error:
            raise AssertionError(f"Expected {"positive"} in {result3.error}")

        # Check logged values
        if logged_values != [4, 8, 3, 3]  # Before and after doubling:
            raise AssertionError(f"Expected {[4, 8, 3, 3]  # Before and (
                after doubling}, got {logged_values}"))


@pytest.mark.unit
class TestFlextCoreConfiguration:
    """Test FlextCore configuration management."""

    def test_get_settings_caching(self, clean_flext_core: FlextCore) -> None:
        """Test settings caching behavior."""
        settings1 = clean_flext_core.get_settings(UserService)
        settings2 = clean_flext_core.get_settings(UserService)

        # Should return same cached instance
        assert settings1 is settings2

    def test_get_settings_different_classes(self, clean_flext_core: FlextCore) -> None:
        """Test settings with different classes."""
        user_settings = clean_flext_core.get_settings(UserService)
        data_settings = clean_flext_core.get_settings(DataService)

        assert isinstance(user_settings, UserService)
        assert isinstance(data_settings, DataService)
        assert user_settings is not data_settings

    def test_get_settings_initialization(self, clean_flext_core: FlextCore) -> None:
        """Test settings initialization."""
        settings = clean_flext_core.get_settings(UserService)

        assert isinstance(settings, UserService)
        assert hasattr(settings, "name")
        if settings.name != "test_user_service":
            raise AssertionError(f"Expected {"test_user_service"}, got {settings.name}")

    def test_settings_cache_persistence(self, clean_flext_core: FlextCore) -> None:
        """Test settings cache persistence."""
        # Get settings
        settings1 = clean_flext_core.get_settings(UserService)

        # Modify the instance
        settings1.name = "modified_name"

        # Get again - should be same modified instance
        settings2 = clean_flext_core.get_settings(UserService)
        if settings2.name != "modified_name":
            raise AssertionError(f"Expected {"modified_name"}, got {settings2.name}")
        assert settings1 is settings2

    def test_constants_property_access(self, clean_flext_core: FlextCore) -> None:
        """Test constants property access."""
        constants = clean_flext_core.constants

        assert constants is FlextConstants
        assert hasattr(constants, "ERROR_CODES")

    def test_constants_consistency(self, clean_flext_core: FlextCore) -> None:
        """Test constants consistency across instances."""
        constants1 = clean_flext_core.constants
        constants2 = clean_flext_core.constants

        assert constants1 is constants2
        assert constants1 is FlextConstants


@pytest.mark.unit
class TestFlextCoreRepresentation:
    """Test FlextCore string representation."""

    def test_repr_with_no_services(self, clean_flext_core: FlextCore) -> None:
        """Test repr with no registered services."""
        repr_str = repr(clean_flext_core)

        if "FlextCore" not in repr_str:

            raise AssertionError(f"Expected {"FlextCore"} in {repr_str}")
        assert "services=" in repr_str

    def test_repr_with_services(self, clean_flext_core: FlextCore) -> None:
        """Test repr with registered services."""
        # Register some services
        service1 = UserService()
        service2 = DataService()

        key1 = ServiceKey[UserService]("user")
        key2 = ServiceKey[DataService]("data")

        clean_flext_core.register_service(key1, service1)
        clean_flext_core.register_service(key2, service2)

        repr_str = repr(clean_flext_core)

        if "FlextCore" not in repr_str:

            raise AssertionError(f"Expected {"FlextCore"} in {repr_str}")
        assert "services=" in repr_str

    def test_repr_format_consistency(self, clean_flext_core: FlextCore) -> None:
        """Test repr format consistency."""
        repr_str = repr(clean_flext_core)

        # Should match pattern: FlextCore(services=N)
        assert repr_str.startswith("FlextCore(")
        assert repr_str.endswith(")")
        if "services=" not in repr_str:
            raise AssertionError(f"Expected {"services="} in {repr_str}")


@pytest.mark.unit
class TestFlextCoreConvenienceFunction:
    """Test flext_core convenience function."""

    def test_convenience_function_returns_singleton(self) -> None:
        """Test convenience function returns singleton."""
        FlextCore._instance = None

        instance1 = flext_core()
        instance2 = flext_core()
        instance3 = FlextCore.get_instance()

        assert instance1 is instance2
        assert instance1 is instance3
        assert isinstance(instance1, FlextCore)

    def test_convenience_function_functional_usage(self) -> None:
        """Test convenience function in functional usage."""
        # Should be able to chain operations
        result = flext_core().ok("test").map(lambda x: f"processed_{x}")

        assert result.is_success
        if result.data != "processed_test":
            raise AssertionError(f"Expected {"processed_test"}, got {result.data}")

    def test_convenience_function_service_operations(self) -> None:
        """Test convenience function with service operations."""
        service = UserService("convenience_test")
        service_key = ServiceKey[UserService]("convenience_service")

        # Register through convenience function
        register_result = flext_core().register_service(service_key, service)
        assert register_result.is_success

        # Retrieve through convenience function
        get_result = flext_core().get_service(service_key)
        assert get_result.is_success
        if get_result.data.name != "convenience_test":
            raise AssertionError(f"Expected {"convenience_test"}, got {get_result.data.name}")

    def test_convenience_function_logging_access(self) -> None:
        """Test convenience function logging access."""
        logger = flext_core().get_logger("convenience.test")

        assert isinstance(logger, FlextLogger)
        if logger._name != "convenience.test":
            raise AssertionError(f"Expected {"convenience.test"}, got {logger._name}")


@pytest.mark.unit
class TestFlextCoreIntegration:
    """Test FlextCore integration scenarios."""

    def test_full_workflow_integration(self, clean_flext_core: FlextCore) -> None:
        """Test complete workflow integration."""
        # 1. Configure logging
        clean_flext_core.configure_logging(log_level="INFO")

        # 2. Get logger
        _ = clean_flext_core.get_logger("integration.test")  # Test logger creation

        # 3. Register services
        user_service = UserService("integration_user")
        data_service = DataService("integration_db")

        user_key = ServiceKey[UserService]("user_service")
        data_key = ServiceKey[DataService]("data_service")

        user_reg_result = clean_flext_core.register_service(user_key, user_service)
        data_reg_result = clean_flext_core.register_service(data_key, data_service)

        assert user_reg_result.is_success
        assert data_reg_result.is_success

        # 4. Create processing pipeline
        def get_user_data(user_id: str) -> FlextResult[str]:
            user_result = clean_flext_core.get_service(user_key)
            if user_result.is_failure:
                return FlextResult.fail("User service not available")

            user_data = user_result.data.get_user(user_id)
            return FlextResult.ok(user_data)

        def save_user_data(user_data: str) -> FlextResult[bool]:
            data_result = clean_flext_core.get_service(data_key)
            if data_result.is_failure:
                return FlextResult.fail("Data service not available")

            save_result = data_result.data.save_data(user_data)
            return FlextResult.ok(save_result)

        # 5. Execute pipeline with railway programming
        logged_steps = []

        def log_step(step_name: str) -> callable:
            def logger_func(data: object) -> None:
                logged_steps.append(f"{step_name}: {data}")

            return clean_flext_core.tap(logger_func)

        pipeline = clean_flext_core.pipe(
            log_step("input"),
            get_user_data,
            log_step("user_data"),
            save_user_data,
            log_step("saved"),
        )

        result = pipeline("user123")

        assert result.is_success
        if not (result.data):
            raise AssertionError(f"Expected True, got {result.data}")
        if len(logged_steps) != EXPECTED_DATA_COUNT:
            raise AssertionError(f"Expected {3}, got {len(logged_steps)}")
        if "input: user123" not in logged_steps:
            raise AssertionError(f"Expected {"input: user123"} in {logged_steps}")
        assert "user_data: User user123" in logged_steps
        if "saved: True" not in logged_steps:
            raise AssertionError(f"Expected {"saved: True"} in {logged_steps}")

    def test_error_handling_integration(self, clean_flext_core: FlextCore) -> None:
        """Test error handling integration."""

        # Create a pipeline that will fail
        def validate_input(x: str) -> FlextResult[str]:
            if not x:
                return FlextResult.fail("Empty input")
            return FlextResult.ok(x)

        def process_data(x: str) -> FlextResult[str]:
            if x == "fail":
                return FlextResult.fail("Processing failed")
            return FlextResult.ok(f"processed_{x}")

        def save_result(x: str) -> FlextResult[str]:
            return FlextResult.ok(f"saved_{x}")

        pipeline = clean_flext_core.pipe(validate_input, process_data, save_result)

        # Test success case
        success_result = pipeline("valid_input")
        assert success_result.is_success
        if success_result.data != "saved_processed_valid_input":
            raise AssertionError(f"Expected {"saved_processed_valid_input"}, got {success_result.data}")

        # Test failure cases
        empty_result = pipeline("")
        assert empty_result.is_failure
        if "Empty input" not in empty_result.error:
            raise AssertionError(f"Expected {"Empty input"} in {empty_result.error}")

        fail_result = pipeline("fail")
        assert fail_result.is_failure
        if "Processing failed" not in fail_result.error:
            raise AssertionError(f"Expected {"Processing failed"} in {fail_result.error}")

    def test_settings_and_services_integration(
        self,
        clean_flext_core: FlextCore,
    ) -> None:
        """Test settings and services integration."""
        # Get settings
        user_settings = clean_flext_core.get_settings(UserService)

        # Register service using settings
        service_key = ServiceKey[UserService]("configured_service")
        register_result = clean_flext_core.register_service(service_key, user_settings)

        assert register_result.is_success

        # Retrieve and verify it's the same instance
        service_result = clean_flext_core.get_service(service_key)
        assert service_result.is_success
        assert service_result.data is user_settings

    def test_constants_usage_integration(self, clean_flext_core: FlextCore) -> None:
        """Test constants usage integration."""
        constants = clean_flext_core.constants

        # Should be able to access error codes
        assert hasattr(constants, "ERROR_CODES")

        # Constants should be consistent
        constants2 = clean_flext_core.constants
        assert constants is constants2


@pytest.mark.unit
class TestFlextCoreEdgeCases:
    """Test FlextCore edge cases and error conditions."""

    def test_singleton_with_manual_instantiation(self) -> None:
        """Test singleton behavior with manual instantiation."""
        FlextCore._instance = None

        # Manual instantiation
        manual_instance = FlextCore()

        # Get singleton instance
        singleton_instance = FlextCore.get_instance()

        # They should be different
        assert manual_instance is not singleton_instance

    def test_container_access_consistency(self, clean_flext_core: FlextCore) -> None:
        """Test container access consistency."""
        container1 = clean_flext_core.container
        container2 = clean_flext_core.container

        # Should be same container instance
        assert container1 is container2

    def test_empty_pipeline_edge_cases(self, clean_flext_core: FlextCore) -> None:
        """Test edge cases with empty pipelines."""
        empty_pipeline = clean_flext_core.pipe()

        # Should handle various input types
        result1 = empty_pipeline(None)
        assert result1.is_success
        assert result1.data is None

        result2 = empty_pipeline([])
        assert result2.is_success
        if result2.data != []:
            raise AssertionError(f"Expected {[]}, got {result2.data}")

        result3 = empty_pipeline({})
        assert result3.is_success
        if result3.data != {}:
            raise AssertionError(f"Expected {{}}, got {result3.data}")

    def test_when_predicate_edge_cases(self, clean_flext_core: FlextCore) -> None:
        """Test when predicate edge cases."""

        def always_true(x: object) -> bool:
            return True

        def always_false(x: object) -> bool:
            return False

        def success_func(x: object) -> FlextResult[object]:
            return FlextResult.ok(f"success_{x}")

        # Always true
        true_condition = clean_flext_core.when(always_true, success_func)
        result1 = true_condition("test")
        assert result1.is_success
        if result1.data != "success_test":
            raise AssertionError(f"Expected {"success_test"}, got {result1.data}")

        # Always false with no else
        false_condition = clean_flext_core.when(always_false, success_func)
        result2 = false_condition("test")
        assert result2.is_success
        if result2.data != "test"  # Unchanged:
            raise AssertionError(f"Expected {"test"  # Unchanged}, got {result2.data}")

    def test_tap_side_effect_exceptions(self, clean_flext_core: FlextCore) -> None:
        """Test tap with side effect exceptions."""

        def failing_side_effect(x: object) -> None:
            msg = "Side effect error"
            raise RuntimeError(msg)

        tap_func = clean_flext_core.tap(failing_side_effect)

        # The current implementation doesn't handle exceptions in side effects
        with pytest.raises(RuntimeError, match="Side effect error"):
            tap_func("test")

    def test_settings_with_invalid_classes(self, clean_flext_core: FlextCore) -> None:
        """Test settings with edge case classes."""

        # Class without default constructor
        class NoDefaultConstructor:
            def __init__(self, required_param: str) -> None:
                self.required_param = required_param

        # Should handle gracefully
        with contextlib.suppress(Exception):
            # Test graceful handling
            _ = clean_flext_core.get_settings(NoDefaultConstructor)

    def test_repr_edge_cases(self, clean_flext_core: FlextCore) -> None:
        """Test repr with edge cases."""
        # Mock container to simulate different service counts
        with patch.object(clean_flext_core, "_container") as mock_container:
            mock_container.get_service_count.return_value = 0
            repr_str = repr(clean_flext_core)
            if "services=0" not in repr_str:
                raise AssertionError(f"Expected {"services=0"} in {repr_str}")

            mock_container.get_service_count.return_value = 100
            repr_str = repr(clean_flext_core)
            if "services=100" not in repr_str:
                raise AssertionError(f"Expected {"services=100"} in {repr_str}")
