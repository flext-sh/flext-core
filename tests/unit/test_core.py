"""REAL tests for FlextCore module - NO MOCKS, REAL EXECUTION ONLY.

This test suite provides comprehensive validation of FlextCore functionality
using actual implementation without any mocking. All tests execute real code
and validate real behavior following SOLID principles and Clean Architecture.
"""

from __future__ import annotations

import contextlib
import io
import time
from datetime import UTC, datetime
from enum import StrEnum

import pytest

from flext_core import (
    FlextCore,
    FlextDomainService,
    FlextEntity,
    FlextEntityId,
    FlextError,
    FlextLogger,
    FlextResult,
    FlextServiceKey,
    ValidationAdapters,
    flext_core,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


# REAL production-style entities using flext-core patterns (minimal implementation for tests)


class UserStatus(StrEnum):
    """Production-style user status enum."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    LOCKED = "locked"


class UserRole(StrEnum):
    """Production-style user role enum."""

    USER = "user"
    ADMIN = "REDACTED_LDAP_BIND_PASSWORD"


class TestUser(FlextEntity):
    """REAL production-style user entity using FlextEntity inheritance.

    This is a minimal production-style implementation within tests/
    that follows real flext-core patterns without importing flext-auth.
    """

    username: str
    email: str
    password_hash: str
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.ACTIVE
    failed_login_attempts: int = 0

    def validate_domain_rules(self) -> FlextResult[None]:
        """REAL domain validation using production business rules."""
        username_result = self._validate_username()
        if username_result.is_failure:
            return username_result

        return self._validate_other_fields()

    def _validate_username(self) -> FlextResult[None]:
        """Validate username field."""
        if not self.username or not self.username.strip():
            return FlextResult[None].fail("Username cannot be empty")
        if len(self.username) < 3:
            return FlextResult[None].fail("Username must be at least 3 characters")
        if len(self.username) > 50:
            return FlextResult[None].fail("Username must be at most 50 characters")
        return FlextResult[None].ok(None)

    def _validate_other_fields(self) -> FlextResult[None]:
        """Validate other required fields."""
        if not self.email or "@" not in self.email:
            return FlextResult[None].fail("Email must contain @ symbol")
        if not self.password_hash:
            return FlextResult[None].fail("Password hash cannot be empty")
        if self.failed_login_attempts < 0:
            return FlextResult[None].fail("Failed login attempts cannot be negative")
        return FlextResult[None].ok(None)

    def is_active(self) -> bool:
        """Check if user is active."""
        return self.status == UserStatus.ACTIVE

    def is_REDACTED_LDAP_BIND_PASSWORD(self) -> bool:
        """Check if user is REDACTED_LDAP_BIND_PASSWORD."""
        return self.role == UserRole.ADMIN


# Real Domain Service using flext-core infrastructure and REAL production entities
class UserManagementService(FlextDomainService[TestUser]):
    """REAL production domain service for user management using production-style entities."""

    def __init__(self) -> None:
        """Initialize with real flext-core foundation."""
        super().__init__()

    def execute(self) -> FlextResult[TestUser]:
        """Execute default user management operation (required by abstract base)."""
        # Default implementation - create a real production user
        return self.create_user_with_validation(
            "usr_default", "default_user", "default@example.com", "hash_sample"
        )

    def create_user_with_validation(
        self, user_id: str, username: str, email: str, password_hash: str
    ) -> FlextResult[TestUser]:
        """Create REAL TestUser with comprehensive production validation."""
        email_validation = FlextCore.validate_email(email)
        if email_validation.is_failure:
            return FlextResult[TestUser].fail(
                f"Invalid email: {email_validation.error}"
            )

        username_validation = self._validate_username_input(username)
        if username_validation.is_failure:
            return FlextResult[TestUser].fail(
                f"Invalid username: {username_validation.error}"
            )

        return self._create_validated_user(user_id, username, email, password_hash)

    def _validate_username_input(self, username: str) -> FlextResult[None]:
        """Validate username input parameters."""
        if not username or not username.strip():
            return FlextResult[None].fail("Invalid username: Username cannot be empty")
        if len(username) < 3:
            return FlextResult[None].fail(
                "Invalid username: Username must be at least 3 characters"
            )
        if len(username) > 50:
            return FlextResult[None].fail(
                "Invalid username: Username must be at most 50 characters"
            )
        return FlextResult[None].ok(None)

    def _create_validated_user(
        self, user_id: str, username: str, email: str, password_hash: str
    ) -> FlextResult[TestUser]:
        """Create and validate TestUser entity."""
        try:
            user = TestUser(
                id=FlextEntityId(user_id),
                username=username,
                email=email,
                password_hash=password_hash,
                role=UserRole.USER,
                status=UserStatus.ACTIVE,
                failed_login_attempts=0,
            )

            validation_result = user.validate_domain_rules()
            if validation_result.is_failure:
                return FlextResult[TestUser].fail(
                    f"Domain validation failed: {validation_result.error}"
                )

            return FlextResult[TestUser].ok(user)
        except Exception as e:
            return FlextResult[TestUser].fail(f"Failed to create user: {e}")

    def validate_user_business_rules(self, user: TestUser) -> FlextResult[None]:
        """Validate using REAL production domain business rules."""
        return user.validate_domain_rules()


# Real Data Repository using flext-core patterns and inheriting from FlextEntity
class UserRepository(FlextEntity):
    """REAL repository using FlextEntity inheritance with production patterns."""

    def __init__(self, repository_id: str = "user_repository") -> None:
        """Initialize repository with FlextEntity inheritance."""
        super().__init__(id=FlextEntityId(repository_id))
        self.storage: dict[str, TestUser] = {}

    def save_user(self, user: TestUser) -> FlextResult[TestUser]:
        """Save user using real entity storage with production validation."""
        validation_result = user.validate_domain_rules()
        if validation_result.is_failure:
            return FlextResult[TestUser].fail(
                f"Validation failed: {validation_result.error}"
            )

        self.storage[str(user.id)] = user
        return FlextResult[TestUser].ok(user)

    def find_user(self, user_id: str) -> FlextResult[TestUser]:
        """Find user with real error handling."""
        if user_id not in self.storage:
            return FlextResult[TestUser].fail(f"User {user_id} not found")
        return FlextResult[TestUser].ok(self.storage[user_id])

    def list_all_users(self) -> FlextResult[list[TestUser]]:
        """List all users with real collection handling."""
        return FlextResult[list[TestUser]].ok(list(self.storage.values()))

    def validate_domain_rules(self) -> FlextResult[None]:
        """Repository domain validation (required by FlextEntity)."""
        if not self.id:
            return FlextResult[None].fail("Repository ID cannot be empty")
        return FlextResult[None].ok(None)


# Test fixtures using real flext-core implementations
@pytest.fixture
def clean_flext_core() -> FlextCore:
    """Create a fresh FlextCore instance for testing."""
    # Reset singleton to ensure clean state
    FlextCore._instance = None
    return FlextCore.get_instance()


@pytest.fixture
def real_user_management_service() -> UserManagementService:
    """Real domain service using flext-core patterns."""
    return UserManagementService()


@pytest.fixture
def real_user_repository() -> UserRepository:
    """Real repository using flext-core entity patterns."""
    return UserRepository()


@pytest.fixture
def user_service_key() -> FlextServiceKey[UserManagementService]:
    """Service key for real user management service."""
    return FlextServiceKey[UserManagementService]("user_management_service")


@pytest.fixture
def repository_service_key() -> FlextServiceKey[UserRepository]:
    """Service key for real user repository."""
    return FlextServiceKey[UserRepository]("user_repository")


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

        # Add some state using string key (type compatible)
        test_key = "test_state_key"
        test_value = {"test": "data"}
        # Use a proper way to store state - using the container instead
        register_result = instance1.register_service(test_key, test_value)
        assert register_result.success

        # Get instance again
        instance2 = FlextCore.get_instance()

        # Verify state persisted
        assert instance1 is instance2
        get_result = instance2.get_service(test_key)
        assert get_result.success
        assert get_result.value == test_value


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
        assert callable(container.register)
        assert callable(container.get)

    def test_register_and_retrieve_real_service(
        self,
        clean_flext_core: FlextCore,
        user_service_key: FlextServiceKey[UserManagementService],
        real_user_management_service: UserManagementService,
    ) -> None:
        """Test registration and retrieval of real domain service."""
        # Register real domain service - convert FlextServiceKey to string
        register_result = clean_flext_core.register_service(
            str(user_service_key), real_user_management_service
        )
        assert register_result.success

        # Retrieve and verify it's the same real service
        retrieval_result = clean_flext_core.get_service(str(user_service_key))
        assert retrieval_result.success
        assert retrieval_result.value is real_user_management_service
        assert isinstance(retrieval_result.value, UserManagementService)
        # Verify it's a real FlextDomainService
        assert isinstance(retrieval_result.value, FlextDomainService)

    def test_real_service_functionality_through_container(
        self,
        clean_flext_core: FlextCore,
        user_service_key: FlextServiceKey[UserManagementService],
        real_user_management_service: UserManagementService,
    ) -> None:
        """Test real domain service functionality through container."""
        # Register real domain service
        clean_flext_core.register_service(
            str(user_service_key), real_user_management_service
        )

        # Retrieve service
        service_result = clean_flext_core.get_service(str(user_service_key))
        assert service_result.success
        service = service_result.value
        assert isinstance(service, UserManagementService)

        # Test real domain functionality with validation
        create_result = service.create_user_with_validation(
            "user1", "john_doe", "john.doe@example.com", "password_hash_123"
        )
        assert create_result.success
        created_user = create_result.value
        assert isinstance(created_user, TestUser)
        assert created_user.username == "john_doe"
        assert created_user.email == "john.doe@example.com"

        # Test real business rule validation
        validation_result = service.validate_user_business_rules(created_user)
        assert validation_result.success

    def test_multiple_real_services(self, clean_flext_core: FlextCore) -> None:
        """Test registration and usage of multiple real flext-core services."""
        user_service = UserManagementService()
        user_repository = UserRepository()

        user_key = FlextServiceKey[UserManagementService]("user_management")
        repo_key = FlextServiceKey[UserRepository]("user_repository")

        # Register both real services
        user_register_result = clean_flext_core.register_service(
            str(user_key), user_service
        )
        repo_register_result = clean_flext_core.register_service(
            str(repo_key), user_repository
        )

        assert user_register_result.success
        assert repo_register_result.success

        # Retrieve and test both services
        user_result = clean_flext_core.get_service(str(user_key))
        repo_result = clean_flext_core.get_service(str(repo_key))

        assert user_result.success
        assert repo_result.success

        # Test real functionality
        user_svc = user_result.value
        repo_svc = repo_result.value
        assert isinstance(user_svc, UserManagementService)
        assert isinstance(repo_svc, UserRepository)

        # Real domain service functionality
        create_user_result = user_svc.create_user_with_validation(
            "test_user", "test_user", "test@example.com", "hash_test_123"
        )
        assert create_user_result.success
        test_user = create_user_result.value

        # Real repository functionality
        save_user_result = repo_svc.save_user(test_user)
        assert save_user_result.success

        # Verify real entity behavior
        assert isinstance(user_svc, FlextDomainService)
        assert isinstance(repo_svc, UserRepository)
        assert repo_svc.id == "user_repository"

    def test_service_not_found_real_error(self, clean_flext_core: FlextCore) -> None:
        """Test real error when service not found."""
        non_existent_key = FlextServiceKey[UserManagementService](
            "non_existent_service"
        )

        result = clean_flext_core.get_service(str(non_existent_key))

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
        assert callable(logger.info)

    def test_configure_logging_real_behavior(self, clean_flext_core: FlextCore) -> None:
        """Test real logging configuration."""
        # Test configuration with real parameters
        clean_flext_core.configure_logging(log_level="DEBUG", _json_output=False)

        # Get logger and verify configuration worked
        logger = clean_flext_core.get_logger("config_test_logger")

        # Verify logger is functional (this is real execution)
        assert isinstance(logger, FlextLogger)

        # Test actual logging (real operation)
        with contextlib.redirect_stderr(io.StringIO()):
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

        # Context manager doesn't have success/data attributes
        context = context_result

        # Verify real context properties
        # The context data is stored in the _context attribute
        assert hasattr(context, "_context")
        assert context._context["operation"] == "test_operation"
        assert context._context["user_id"] == "test_user"
        assert context._context["correlation_id"] == "test_correlation"


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

        sequence_result = clean_flext_core.sequence(
            [
                FlextResult[object].ok(r.value)
                if r.success
                else FlextResult[object].fail(r.error or "Error")
                for r in results
            ]
        )

        assert sequence_result.success
        assert sequence_result.value == [1, 2, 3]

    def test_sequence_with_real_failure(self, clean_flext_core: FlextCore) -> None:
        """Test sequence operation with real failure."""
        # Mix success and failure results
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("Real error occurred"),
            FlextResult[int].ok(3),
        ]

        sequence_result = clean_flext_core.sequence(
            [
                FlextResult[object].ok(r.value)
                if r.success
                else FlextResult[object].fail(r.error or "Error")
                for r in results
            ]
        )

        assert sequence_result.is_failure
        assert sequence_result.error is not None
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

        first_success_result = clean_flext_core.first_success(
            [
                FlextResult[object].ok(r.value)
                if r.success
                else FlextResult[object].fail(r.error or "Error")
                for r in results
            ]
        )

        assert first_success_result.success
        assert first_success_result.value == "Success value"


@pytest.mark.unit
class TestFlextCoreValidationIntegration:
    """Test FlextCore validation integration with real validators."""

    def test_validate_email_real_functionality(
        self, clean_flext_core: FlextCore
    ) -> None:
        """Test real email validation."""
        # Test valid email
        valid_result = clean_flext_core.validate_email("test@example.com")
        assert valid_result.success

        # Test invalid email
        invalid_result = clean_flext_core.validate_email("invalid-email")
        assert invalid_result.is_failure

    def test_validate_string_real_behavior(self, clean_flext_core: FlextCore) -> None:
        """Test real string validation."""
        # Test non-empty string
        valid_result = clean_flext_core.validate_string("test string")
        assert valid_result.success

        # Test empty string using require_non_empty
        empty_result = clean_flext_core.require_non_empty("")
        assert empty_result.is_failure

        # Test whitespace-only string using require_non_empty
        whitespace_result = clean_flext_core.require_non_empty("   ")
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
            error_msg = "Real error in function"
            raise ValueError(error_msg)

        # Test successful function
        success_result = clean_flext_core.safe_call(successful_function, "default")
        assert success_result == "success_result"

        # Test failing function
        failure_result = clean_flext_core.safe_call(failing_function, "default")
        assert failure_result == "default"

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

    def test_track_performance_real_execution(
        self, clean_flext_core: FlextCore
    ) -> None:
        """Test real performance tracking."""
        # Test that the track_performance method exists and is callable
        assert hasattr(clean_flext_core, "track_performance")
        assert callable(clean_flext_core.track_performance)

        # Create a simple function for testing
        def test_operation() -> str:
            time.sleep(0.01)  # Real delay
            return "operation_result"

        # Apply the performance decorator - cast to callable since track_performance returns object
        decorator = clean_flext_core.track_performance("test_category")
        assert callable(decorator)

        decorated_func = decorator(test_operation)
        assert callable(decorated_func)

        # Execute real operation with tracking
        result = decorated_func()
        assert result == "operation_result"

        # Note: Performance metrics methods not yet implemented in FlextCore
        # This test verifies the decorator works, metrics collection is pending

    def test_performance_tracking_placeholder(
        self, clean_flext_core: FlextCore
    ) -> None:
        """Test performance tracking infrastructure exists."""
        # Note: Performance metrics methods not yet implemented in FlextCore
        # This test verifies the core instance is functional
        assert clean_flext_core is not None

        # Verify the track_performance decorator method exists
        assert hasattr(clean_flext_core, "track_performance")
        assert callable(clean_flext_core.track_performance)


@pytest.mark.unit
class TestFlextCoreEntityCreation:
    """Test FlextCore entity creation with real domain objects."""

    def test_create_entity_real_validation(self) -> None:
        """Test real entity creation with REAL production entities."""
        # Create REAL production entity directly
        entity = TestUser(
            id=FlextEntityId("test_entity_1"),
            username="test_entity",
            email="test@entity.com",
            password_hash="hash_test_123",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
        )

        assert isinstance(entity, TestUser)
        assert isinstance(entity, FlextEntity)
        assert entity.username == "test_entity"
        assert entity.email == "test@entity.com"

        # Test real entity validation
        validation_result = entity.validate_domain_rules()
        assert validation_result.success

        # Test another entity with different status
        direct_entity = TestUser(
            id=FlextEntityId("test_direct"),
            username="direct_test",
            email="direct@test.com",
            password_hash="hash_direct_123",
            role=UserRole.USER,
            status=UserStatus.INACTIVE,
        )

        assert isinstance(direct_entity, TestUser)
        assert direct_entity.status == UserStatus.INACTIVE

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
        metadata = metadata_result.value
        # FlextMetadata has a root attribute and get method
        assert hasattr(metadata, "root")
        assert hasattr(metadata, "get")
        assert metadata.get("source") == "test_source"
        assert metadata.get("version") == "1.0.0"


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
            "Real validation failed"
        )

        # Check the dynamically created class using class name and inheritance
        assert validation_error.__class__.__name__ == "FlextValidationError"
        assert isinstance(validation_error, ValueError)  # Base class
        assert "Real validation failed" in str(validation_error)

    def test_handle_error_real_processing(self) -> None:
        """Test real error handling."""
        real_error = ValueError("Real runtime error")

        handled_result = FlextResult[str].fail(str(real_error))

        assert handled_result.is_failure
        assert handled_result.error
        assert "Real runtime error" in handled_result.error


@pytest.mark.unit
class TestFlextCoreIntegrationScenarios:
    """Test complete integration scenarios with real workflows."""

    def test_complete_workflow_user_management(
        self, clean_flext_core: FlextCore
    ) -> None:
        """Test complete user management workflow with real flext-core operations."""
        # 1. Register real domain services
        user_service = UserManagementService()
        user_repository = UserRepository()

        user_key = FlextServiceKey[UserManagementService]("workflow_user_service")
        repo_key = FlextServiceKey[UserRepository]("workflow_repository")

        # Register real services using flext-core container
        user_reg_result = clean_flext_core.register_service(str(user_key), user_service)
        repo_reg_result = clean_flext_core.register_service(
            str(repo_key), user_repository
        )

        assert user_reg_result.success
        assert repo_reg_result.success

        # 2. Execute real business workflow with domain services
        user_svc_result = clean_flext_core.get_service(str(user_key))
        repo_svc_result = clean_flext_core.get_service(str(repo_key))

        assert user_svc_result.success
        assert repo_svc_result.success

        user_svc = user_svc_result.value
        repo_svc = repo_svc_result.value
        assert isinstance(user_svc, UserManagementService)
        assert isinstance(repo_svc, UserRepository)

        # 3. Real user creation workflow with domain validation
        create_result = user_svc.create_user_with_validation(
            "workflow_user",
            "john_workflow",
            "john.workflow@example.com",
            "hash_workflow_123",
        )
        assert create_result.success
        workflow_user = create_result.value
        assert isinstance(workflow_user, TestUser)

        # 4. Real domain persistence workflow using repository
        save_result = repo_svc.save_user(workflow_user)
        assert save_result.success

        # 5. Real validation workflow using flext-core validators
        email_validation = clean_flext_core.validate_email("john.workflow@example.com")
        assert email_validation.success

        # 6. Real retrieval and verification workflow
        get_user_result = repo_svc.find_user("workflow_user")
        assert get_user_result.success
        retrieved_user = get_user_result.value
        assert isinstance(retrieved_user, TestUser)
        assert retrieved_user.username == "john_workflow"
        assert retrieved_user.email == "john.workflow@example.com"

        # 7. Real business rules validation
        business_validation = user_svc.validate_user_business_rules(retrieved_user)
        assert business_validation.success

    def test_error_recovery_workflow(self, clean_flext_core: FlextCore) -> None:
        """Test real error recovery workflow with domain validation."""
        user_service = UserManagementService()
        user_repository = UserRepository()

        user_key = FlextServiceKey[UserManagementService]("error_recovery_service")
        repo_key = FlextServiceKey[UserRepository]("error_recovery_repo")

        # Register real services
        clean_flext_core.register_service(str(user_key), user_service)
        clean_flext_core.register_service(str(repo_key), user_repository)

        # Get services
        service_result = clean_flext_core.get_service(str(user_key))
        repo_result = clean_flext_core.get_service(str(repo_key))
        assert service_result.success
        assert repo_result.success

        service = service_result.value
        repository = repo_result.value
        assert isinstance(service, UserManagementService)
        assert isinstance(repository, UserRepository)

        # 1. Create user successfully with real validation
        create_result = service.create_user_with_validation(
            "test_user", "test_user", "test@user.com", "hash_test_123"
        )
        assert create_result.success
        user = create_result.value

        save_result = repository.save_user(user)
        assert save_result.success

        # 2. Try to create user with invalid email (real domain validation error)
        invalid_email_result = service.create_user_with_validation(
            "test_user_2", "test_user_2", "invalid-email", "hash_test_456"
        )
        assert invalid_email_result.is_failure
        assert invalid_email_result.error is not None
        assert "Invalid email" in invalid_email_result.error

        # 3. Recovery: create user with valid data
        recovery_result = service.create_user_with_validation(
            "test_user_2", "test_user_2", "test2@user.com", "hash_test_456"
        )
        assert recovery_result.success
        recovery_user = recovery_result.value

        recovery_save = repository.save_user(recovery_user)
        assert recovery_save.success

        # 4. Verify state consistency using real repository
        all_users_result = repository.list_all_users()
        assert all_users_result.success
        all_users = all_users_result.value
        assert len(all_users) == 2

        user_ids = [u.id for u in all_users]
        assert "test_user" in user_ids
        assert "test_user_2" in user_ids


@pytest.mark.unit
class TestFlextCoreGlobalInstance:
    """Test FlextCore global instance functionality."""

    def test_global_instance_access(self) -> None:
        """Test global flext_core function access."""
        # Reset to ensure clean state
        FlextCore._instance = None

        # Access global instance through function call
        global_instance = flext_core()

        assert isinstance(global_instance, FlextCore)
        assert global_instance is FlextCore.get_instance()

    def test_global_instance_functionality(self) -> None:
        """Test global instance has full functionality."""
        # Reset singleton
        FlextCore._instance = None

        # Test functionality through global instance
        uuid_result = flext_core().generate_uuid()
        assert isinstance(uuid_result, str)
        assert len(uuid_result) == 36

        email_validation = flext_core().validate_email("test@global.com")
        assert email_validation.success

        # Test service registration through global instance using real service
        test_service = UserManagementService()
        test_key = FlextServiceKey[UserManagementService]("global_test")

        register_result = flext_core().register_service(str(test_key), test_service)
        assert register_result.success

        get_result = flext_core().get_service(str(test_key))
        assert get_result.success
        assert get_result.value is test_service
        assert isinstance(get_result.value, UserManagementService)
