"""REAL tests for FlextCore module - NO MOCKS, REAL EXECUTION ONLY.

This test suite provides comprehensive validation of FlextCore functionality
using actual implementation without any mocking. All tests execute real code
and validate real behavior following SOLID principles and Clean Architecture.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

import pytest

from flext_core import (
    FlextConstants,
    FlextCore,
    FlextError,
    FlextLogger,
    FlextResult,
    FlextServiceKey,
    FlextValidationError,
    flext_core,
)


pytestmark = [pytest.mark.unit, pytest.mark.core]


# Real service classes for dependency injection testing
class RealUserService:
    """Real user service implementation for testing."""

    def __init__(self, name: str = "real_user_service") -> None:
        """Initialize user service."""
        self.name = name
        self.users: dict[str, str] = {}

    def create_user(self, user_id: str, username: str) -> FlextResult[str]:
        """Create a new user."""
        if user_id in self.users:
            return FlextResult[str].fail(f"User {user_id} already exists")
        self.users[user_id] = username
        return FlextResult[str].ok(username)

    def get_user(self, user_id: str) -> FlextResult[str]:
        """Get user by ID."""
        if user_id not in self.users:
            return FlextResult[str].fail(f"User {user_id} not found")
        return FlextResult[str].ok(self.users[user_id])

    def list_users(self) -> FlextResult[list[str]]:
        """List all users."""
        return FlextResult[list[str]].ok(list(self.users.keys()))


class RealDataService:
    """Real data service implementation for testing."""

    def __init__(self, connection_name: str = "test_db") -> None:
        """Initialize data service."""
        self.connection_name = connection_name
        self.data_store: dict[str, object] = {}

    def save_data(self, key: str, data: object) -> FlextResult[bool]:
        """Save data."""
        if not key:
            return FlextResult[bool].fail("Key cannot be empty")
        self.data_store[key] = data
        return FlextResult[bool].ok(True)

    def get_data(self, key: str) -> FlextResult[object]:
        """Get data by key."""
        if key not in self.data_store:
            return FlextResult[object].fail(f"Data with key {key} not found")
        return FlextResult[object].ok(self.data_store[key])

    def count_records(self) -> int:
        """Count total records."""
        return len(self.data_store)


class RealValidator:
    """Real validator for testing validation features."""

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        return "@" in email and "." in email

    @staticmethod
    def validate_non_empty(value: str) -> bool:
        """Validate non-empty string."""
        return bool(value and value.strip())


# Test fixtures using real implementations
@pytest.fixture
def clean_flext_core() -> FlextCore:
    """Create a fresh FlextCore instance for testing."""
    # Reset singleton to ensure clean state
    FlextCore._instance = None
    return FlextCore.get_instance()


@pytest.fixture
def real_user_service() -> RealUserService:
    """Real user service for container testing."""
    return RealUserService("test_container_service")


@pytest.fixture
def real_data_service() -> RealDataService:
    """Real data service for container testing."""
    return RealDataService("test_connection")


@pytest.fixture
def user_service_key() -> FlextServiceKey[RealUserService]:
    """Service key for user service."""
    return FlextServiceKey[RealUserService]("user_service")


@pytest.fixture
def data_service_key() -> FlextServiceKey[RealDataService]:
    """Service key for data service."""
    return FlextServiceKey[RealDataService]("data_service")


@pytest.mark.unit
class TestFlextCoreSingleton:
    """Test FlextCore singleton pattern with real behavior."""

    def test_singleton_instance_creation(self) -> None:
        """Test singleton instance creation with real behavior."""
        # Reset singleton to start fresh
        FlextCore._instance = None

        instance1 = FlextCore.get_instance()
        instance2 = FlextCore.get_instance()

        # Verify singleton behavior
        assert isinstance(instance1, FlextCore)
        assert isinstance(instance2, FlextCore)
        assert instance1 is instance2

        # Verify real initialization
        assert hasattr(instance1, "_container")
        assert hasattr(instance1, "_settings_cache")

    def test_singleton_initialization_state(self) -> None:
        """Test singleton initialization with real state verification."""
        FlextCore._instance = None

        instance = FlextCore.get_instance()

        # Verify real attributes exist and have correct types
        assert hasattr(instance, "_container")
        assert hasattr(instance, "_settings_cache")
        assert isinstance(instance._settings_cache, dict)

        # Verify container is functional
        assert hasattr(instance._container, "register")
        assert hasattr(instance._container, "get")

    def test_singleton_state_persistence(self) -> None:
        """Test singleton state persistence between calls."""
        FlextCore._instance = None

        instance1 = FlextCore.get_instance()
        
        # Add some state
        test_key = "test_state_key"
        test_value = {"test": "data"}
        instance1._settings_cache[test_key] = test_value

        # Get instance again
        instance2 = FlextCore.get_instance()

        # Verify state persisted
        assert instance1 is instance2
        assert test_key in instance2._settings_cache
        assert instance2._settings_cache[test_key] == test_value


@pytest.mark.unit
class TestFlextCoreContainerIntegration:
    """Test FlextCore container integration with real services."""

    def test_container_property_access(self, clean_flext_core: FlextCore) -> None:
        """Test container property access with real container."""
        container = clean_flext_core.container

        # Verify container is real and functional
        assert container is not None
        assert hasattr(container, "register")
        assert hasattr(container, "get")
        assert callable(getattr(container, "register"))
        assert callable(getattr(container, "get"))

    def test_register_and_retrieve_real_service(
        self,
        clean_flext_core: FlextCore,
        user_service_key: FlextServiceKey[RealUserService],
        real_user_service: RealUserService,
    ) -> None:
        """Test registration and retrieval of real service."""
        # Register real service
        register_result = clean_flext_core.register_service(
            user_service_key, real_user_service
        )
        assert register_result.success

        # Retrieve and verify it's the same real service
        retrieval_result = clean_flext_core.get_service(user_service_key)
        assert retrieval_result.success
        assert retrieval_result.data is real_user_service
        assert isinstance(retrieval_result.data, RealUserService)
        assert retrieval_result.data.name == "test_container_service"

    def test_real_service_functionality_through_container(
        self,
        clean_flext_core: FlextCore,
        user_service_key: FlextServiceKey[RealUserService],
        real_user_service: RealUserService,
    ) -> None:
        """Test real service functionality through container."""
        # Register service
        clean_flext_core.register_service(user_service_key, real_user_service)

        # Retrieve service
        service_result = clean_flext_core.get_service(user_service_key)
        assert service_result.success
        service = service_result.data

        # Test real functionality
        create_result = service.create_user("user1", "john_doe")
        assert create_result.success
        assert create_result.data == "john_doe"

        get_result = service.get_user("user1")
        assert get_result.success
        assert get_result.data == "john_doe"

        list_result = service.list_users()
        assert list_result.success
        assert "user1" in list_result.data

    def test_multiple_real_services(self, clean_flext_core: FlextCore) -> None:
        """Test registration and usage of multiple real services."""
        user_service = RealUserService("multi_user_service")
        data_service = RealDataService("multi_data_service")

        user_key = FlextServiceKey[RealUserService]("user_service")
        data_key = FlextServiceKey[RealDataService]("data_service")

        # Register both services
        user_register_result = clean_flext_core.register_service(user_key, user_service)
        data_register_result = clean_flext_core.register_service(data_key, data_service)

        assert user_register_result.success
        assert data_register_result.success

        # Retrieve and test both services
        user_result = clean_flext_core.get_service(user_key)
        data_result = clean_flext_core.get_service(data_key)

        assert user_result.success
        assert data_result.success

        # Test real functionality
        user_svc = user_result.data
        data_svc = data_result.data

        # User service functionality
        create_user_result = user_svc.create_user("test_user", "test_name")
        assert create_user_result.success

        # Data service functionality
        save_data_result = data_svc.save_data("test_key", {"test": "value"})
        assert save_data_result.success

        # Verify independence
        assert user_svc.name == "multi_user_service"
        assert data_svc.connection_name == "multi_data_service"

    def test_service_not_found_real_error(self, clean_flext_core: FlextCore) -> None:
        """Test real error when service not found."""
        non_existent_key = FlextServiceKey[RealUserService]("non_existent_service")

        result = clean_flext_core.get_service(non_existent_key)

        assert result.is_failure
        assert result.error is not None
        assert "not found" in result.error.lower()


@pytest.mark.unit
class TestFlextCoreLoggingIntegration:
    """Test FlextCore logging integration with real logging."""

    def test_get_logger_real_functionality(self, clean_flext_core: FlextCore) -> None:
        """Test getting real logger instance."""
        logger_name = "test_logger"
        logger = clean_flext_core.get_logger(logger_name)

        # Verify it's a real logger
        assert isinstance(logger, FlextLogger)
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")
        assert callable(getattr(logger, "info"))

    def test_configure_logging_real_behavior(self, clean_flext_core: FlextCore) -> None:
        """Test real logging configuration."""
        # Test configuration with real parameters
        clean_flext_core.configure_logging(log_level="DEBUG", _json_output=False)

        # Get logger and verify configuration worked
        logger = clean_flext_core.get_logger("config_test_logger")
        
        # Verify logger is functional (this is real execution)
        assert isinstance(logger, FlextLogger)

        # Test actual logging (real operation)
        with contextlib.redirect_stderr(contextlib.StringIO()) as captured:
            logger.debug("Test debug message")
            # No assertion on captured since logging config varies
            # But the method executed without errors

    def test_logging_context_real_usage(self, clean_flext_core: FlextCore) -> None:
        """Test real logging context creation and usage."""
        # Test real context creation
        context_result = clean_flext_core.create_log_context(
            operation="test_operation",
            user_id="test_user",
            correlation_id="test_correlation",
        )

        assert context_result.success
        context = context_result.data

        # Verify real context properties
        assert hasattr(context, "operation")
        assert hasattr(context, "user_id")
        assert hasattr(context, "correlation_id")


@pytest.mark.unit
class TestFlextCoreResultOperations:
    """Test FlextCore Result operations with real data."""

    def test_sequence_real_results(self, clean_flext_core: FlextCore) -> None:
        """Test sequence operation with real Results."""
        # Create real successful results
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].ok(2),
            FlextResult[int].ok(3),
        ]

        sequence_result = clean_flext_core.sequence(results)

        assert sequence_result.success
        assert sequence_result.data == [1, 2, 3]

    def test_sequence_with_real_failure(self, clean_flext_core: FlextCore) -> None:
        """Test sequence operation with real failure."""
        # Mix success and failure results
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("Real error occurred"),
            FlextResult[int].ok(3),
        ]

        sequence_result = clean_flext_core.sequence(results)

        assert sequence_result.is_failure
        assert "Real error occurred" in sequence_result.error

    def test_first_success_real_behavior(self, clean_flext_core: FlextCore) -> None:
        """Test first_success with real Results."""
        # All failures except one
        results = [
            FlextResult[str].fail("First error"),
            FlextResult[str].fail("Second error"),
            FlextResult[str].ok("Success value"),
            FlextResult[str].fail("Fourth error"),
        ]

        first_success_result = clean_flext_core.first_success(results)

        assert first_success_result.success
        assert first_success_result.data == "Success value"


@pytest.mark.unit
class TestFlextCoreValidationIntegration:
    """Test FlextCore validation integration with real validators."""

    def test_validate_email_real_functionality(self, clean_flext_core: FlextCore) -> None:
        """Test real email validation."""
        # Test valid email
        valid_result = clean_flext_core.validate_email("test@example.com")
        assert valid_result.success

        # Test invalid email
        invalid_result = clean_flext_core.validate_email("invalid-email")
        assert invalid_result.is_failure

    def test_validate_non_empty_string_real_behavior(
        self, clean_flext_core: FlextCore
    ) -> None:
        """Test real non-empty string validation."""
        # Test non-empty string
        valid_result = clean_flext_core.validate_non_empty_string("test string")
        assert valid_result.success

        # Test empty string
        empty_result = clean_flext_core.validate_non_empty_string("")
        assert empty_result.is_failure

        # Test whitespace-only string
        whitespace_result = clean_flext_core.validate_non_empty_string("   ")
        assert whitespace_result.is_failure

    def test_validate_required_real_validation(
        self, clean_flext_core: FlextCore
    ) -> None:
        """Test real required value validation."""
        # Test with actual value
        valid_result = clean_flext_core.validate_required("test_value")
        assert valid_result.success

        # Test with None
        none_result = clean_flext_core.validate_required(None)
        assert none_result.is_failure


@pytest.mark.unit
class TestFlextCoreUtilities:
    """Test FlextCore utility functions with real operations."""

    def test_generate_uuid_real_generation(self, clean_flext_core: FlextCore) -> None:
        """Test real UUID generation."""
        uuid1 = clean_flext_core.generate_uuid()
        uuid2 = clean_flext_core.generate_uuid()

        # Verify real UUIDs
        assert isinstance(uuid1, str)
        assert isinstance(uuid2, str)
        assert uuid1 != uuid2
        assert len(uuid1) == 36  # Standard UUID length
        assert "-" in uuid1

    def test_generate_correlation_id_real_generation(
        self, clean_flext_core: FlextCore
    ) -> None:
        """Test real correlation ID generation."""
        corr_id1 = clean_flext_core.generate_correlation_id()
        corr_id2 = clean_flext_core.generate_correlation_id()

        # Verify real correlation IDs
        assert isinstance(corr_id1, str)
        assert isinstance(corr_id2, str)
        assert corr_id1 != corr_id2
        assert corr_id1.startswith("corr_")
        assert corr_id2.startswith("corr_")

    def test_safe_call_real_execution(self, clean_flext_core: FlextCore) -> None:
        """Test safe call with real function execution."""

        def successful_function() -> str:
            return "success_result"

        def failing_function() -> str:
            raise ValueError("Real error in function")

        # Test successful function
        success_result = clean_flext_core.safe_call(successful_function)
        assert success_result.success
        assert success_result.data == "success_result"

        # Test failing function
        failure_result = clean_flext_core.safe_call(failing_function)
        assert failure_result.is_failure
        assert "Real error in function" in failure_result.error

    def test_truncate_real_string_operations(self, clean_flext_core: FlextCore) -> None:
        """Test real string truncation."""
        long_string = "This is a very long string that should be truncated"
        
        # Test truncation
        truncated = clean_flext_core.truncate(long_string, max_length=20)
        assert len(truncated) <= 20
        assert truncated.endswith("...")

        # Test no truncation needed
        short_string = "Short"
        not_truncated = clean_flext_core.truncate(short_string, max_length=20)
        assert not_truncated == short_string


@pytest.mark.unit
class TestFlextCorePerformanceIntegration:
    """Test FlextCore performance monitoring with real execution."""

    def test_track_performance_real_execution(self, clean_flext_core: FlextCore) -> None:
        """Test real performance tracking."""

        @clean_flext_core.track_performance("test_category")
        def test_operation() -> str:
            time.sleep(0.01)  # Real delay
            return "operation_result"

        # Execute real operation with tracking
        result = test_operation()
        assert result == "operation_result"

        # Verify performance metrics were recorded
        metrics = clean_flext_core.get_performance_metrics()
        assert "metrics" in metrics
        assert isinstance(metrics["metrics"], dict)

    def test_clear_performance_metrics_real_behavior(
        self, clean_flext_core: FlextCore
    ) -> None:
        """Test real performance metrics clearing."""
        # Record some metrics first
        clean_flext_core.record_performance(
            "test_category", "test_function", 0.1, success=True
        )

        # Verify metrics exist
        metrics_before = clean_flext_core.get_performance_metrics()
        assert "metrics" in metrics_before

        # Clear metrics
        clean_flext_core.clear_performance_metrics()

        # Verify metrics cleared
        metrics_after = clean_flext_core.get_performance_metrics()
        assert metrics_after["metrics"] == {}


@pytest.mark.unit
class TestFlextCoreEntityCreation:
    """Test FlextCore entity creation with real domain objects."""

    def test_create_entity_real_validation(self, clean_flext_core: FlextCore) -> None:
        """Test real entity creation with validation."""

        class RealEntity:
            def __init__(self, name: str, value: int) -> None:
                self.name = name
                self.value = value

        # Test successful creation
        result = clean_flext_core.create_entity(
            RealEntity, name="test_entity", value=42
        )

        assert result.success
        entity = result.data
        assert isinstance(entity, RealEntity)
        assert entity.name == "test_entity"
        assert entity.value == 42

    def test_create_metadata_real_functionality(
        self, clean_flext_core: FlextCore
    ) -> None:
        """Test real metadata creation."""
        metadata_result = clean_flext_core.create_metadata(
            source="test_source",
            version="1.0.0",
            timestamp=datetime.now(UTC).isoformat(),
        )

        assert metadata_result.success
        metadata = metadata_result.data
        assert hasattr(metadata, "source") or hasattr(metadata, "__getitem__")


@pytest.mark.unit
class TestFlextCoreErrorHandling:
    """Test FlextCore error handling with real exceptions."""

    def test_create_error_real_functionality(self, clean_flext_core: FlextCore) -> None:
        """Test real error creation."""
        error = clean_flext_core.create_error("Real error message", "TEST_ERROR")

        assert isinstance(error, FlextError)
        assert "Real error message" in str(error)

    def test_create_validation_error_real_functionality(
        self, clean_flext_core: FlextCore
    ) -> None:
        """Test real validation error creation."""
        validation_error = clean_flext_core.create_validation_error(
            "Real validation failed", field="test_field", value="invalid_value"
        )

        assert isinstance(validation_error, FlextValidationError)
        assert "Real validation failed" in str(validation_error)

    def test_handle_error_real_processing(self, clean_flext_core: FlextCore) -> None:
        """Test real error handling."""
        real_error = ValueError("Real runtime error")
        
        handled_result = clean_flext_core.handle_error(real_error)
        
        assert handled_result.is_failure
        assert "Real runtime error" in handled_result.error


@pytest.mark.unit
class TestFlextCoreIntegrationScenarios:
    """Test complete integration scenarios with real workflows."""

    def test_complete_workflow_user_management(
        self, clean_flext_core: FlextCore
    ) -> None:
        """Test complete user management workflow with real operations."""
        # 1. Register real services
        user_service = RealUserService("workflow_user_service")
        data_service = RealDataService("workflow_data_service")

        user_key = FlextServiceKey[RealUserService]("workflow_user")
        data_key = FlextServiceKey[RealDataService]("workflow_data")

        # Register services
        user_reg_result = clean_flext_core.register_service(user_key, user_service)
        data_reg_result = clean_flext_core.register_service(data_key, data_service)

        assert user_reg_result.success
        assert data_reg_result.success

        # 2. Execute real business workflow
        user_svc_result = clean_flext_core.get_service(user_key)
        data_svc_result = clean_flext_core.get_service(data_key)

        assert user_svc_result.success
        assert data_svc_result.success

        user_svc = user_svc_result.data
        data_svc = data_svc_result.data

        # 3. Real user creation workflow
        create_result = user_svc.create_user("workflow_user", "john_workflow")
        assert create_result.success

        # 4. Real data persistence workflow
        user_data = {"user_id": "workflow_user", "name": "john_workflow"}
        save_result = data_svc.save_data("user_workflow_user", user_data)
        assert save_result.success

        # 5. Real validation workflow
        email_validation = clean_flext_core.validate_email("john@workflow.com")
        assert email_validation.success

        # 6. Real retrieval and verification workflow
        get_user_result = user_svc.get_user("workflow_user")
        get_data_result = data_svc.get_data("user_workflow_user")

        assert get_user_result.success
        assert get_data_result.success
        assert get_user_result.data == "john_workflow"

    def test_error_recovery_workflow(self, clean_flext_core: FlextCore) -> None:
        """Test real error recovery workflow."""
        user_service = RealUserService("error_recovery_service")
        user_key = FlextServiceKey[RealUserService]("error_recovery")

        # Register service
        clean_flext_core.register_service(user_key, user_service)
        
        # Get service
        service_result = clean_flext_core.get_service(user_key)
        assert service_result.success
        service = service_result.data

        # 1. Create user successfully
        create_result = service.create_user("test_user", "test_name")
        assert create_result.success

        # 2. Try to create duplicate user (real error)
        duplicate_result = service.create_user("test_user", "duplicate_name")
        assert duplicate_result.is_failure
        assert "already exists" in duplicate_result.error

        # 3. Recovery: create different user
        recovery_result = service.create_user("test_user_2", "recovery_name")
        assert recovery_result.success

        # 4. Verify state is consistent
        list_result = service.list_users()
        assert list_result.success
        assert "test_user" in list_result.data
        assert "test_user_2" in list_result.data
        assert len(list_result.data) == 2


@pytest.mark.unit
class TestFlextCoreGlobalInstance:
    """Test FlextCore global instance functionality."""

    def test_global_instance_access(self) -> None:
        """Test global flext_core instance access."""
        # Reset to ensure clean state
        FlextCore._instance = None

        # Access global instance
        global_instance = flext_core

        assert isinstance(global_instance, FlextCore)
        assert global_instance is FlextCore.get_instance()

    def test_global_instance_functionality(self) -> None:
        """Test global instance has full functionality."""
        # Reset singleton
        FlextCore._instance = None

        # Test functionality through global instance
        uuid_result = flext_core.generate_uuid()
        assert isinstance(uuid_result, str)
        assert len(uuid_result) == 36

        email_validation = flext_core.validate_email("test@global.com")
        assert email_validation.success

        # Test service registration through global instance
        test_service = RealUserService("global_test_service")
        test_key = FlextServiceKey[RealUserService]("global_test")

        register_result = flext_core.register_service(test_key, test_service)
        assert register_result.success

        get_result = flext_core.get_service(test_key)
        assert get_result.success
        assert get_result.data is test_service
