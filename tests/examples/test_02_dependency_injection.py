# mypy: ignore-errors
"""Unit tests for 02_dependency_injection.py example.

Tests the refactored dependency injection implementation using real FlextContainer
and FlextResult patterns without mocks, following FLEXT testing standards.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from io import StringIO
from pathlib import Path
from typing import cast

import pytest

from flext_core import FlextContainer, FlextResult

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent / "../../examples"))

# Import the example module directly
example_path = Path(__file__).parent / "../../examples/02_dependency_injection.py"
spec = importlib.util.spec_from_file_location("di_example", example_path)
if spec is None or spec.loader is None:
    msg = f"Could not load module from {example_path}"
    raise ImportError(msg)
di_example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(di_example)

# Import the classes we need
UserData = di_example.UserData
User = di_example.User
UserService = di_example.UserService
NotificationService = di_example.NotificationService
UserRegistrationService = di_example.UserRegistrationService
setup_container = di_example.setup_container
main = di_example.main


# =============================================================================
# TEST DOMAIN MODELS
# =============================================================================


class TestDomainModels:
    """Test domain model classes."""

    def test_user_data_creation(self) -> None:
        """Test UserData value object creation."""
        user_data = UserData(name="John Doe", email="john@example.com", age=30)

        assert user_data.name == "John Doe"
        assert user_data.email == "john@example.com"
        assert user_data.age == 30

    def test_user_entity_creation(self) -> None:
        """Test User entity creation with defaults."""
        user = User(
            id="test-id",
            name="Jane Doe",
            email="jane@example.com",
            age=25,
        )

        assert user.id == "test-id"
        assert user.name == "Jane Doe"
        assert user.email == "jane@example.com"
        assert user.age == 25
        assert user.status == "active"

    def test_user_validation_valid(self) -> None:
        """Test user validation with valid data."""
        user = User(
            id="test-id",
            name="Valid Name",
            email="valid@example.com",
            age=30,
        )

        result = user.validate_business_rules()
        assert result.success
        assert result.unwrap() is None

    def test_user_validation_short_name(self) -> None:
        """Test user validation with short name."""
        user = User(
            id="test-id",
            name="A",
            email="valid@example.com",
            age=30,
        )

        result = user.validate_business_rules()
        assert result.is_failure
        assert "Name must be at least 2 characters" in str(result.error)

    def test_user_validation_invalid_email_no_at(self) -> None:
        """Test user validation with email missing @."""
        user = User(
            id="test-id",
            name="Valid Name",
            email="invalid-email",
            age=30,
        )

        result = user.validate_business_rules()
        assert result.is_failure
        assert "Invalid email format" in str(result.error)

    def test_user_validation_invalid_email_no_domain(self) -> None:
        """Test user validation with email missing domain."""
        user = User(
            id="test-id",
            name="Valid Name",
            email="invalid@",
            age=30,
        )

        result = user.validate_business_rules()
        assert result.is_failure
        assert "Invalid email format" in str(result.error)

    def test_user_validation_negative_age(self) -> None:
        """Test user validation with negative age."""
        user = User(
            id="test-id",
            name="Valid Name",
            email="valid@example.com",
            age=-1,
        )

        result = user.validate_business_rules()
        assert result.is_failure
        assert "Age must be between 0 and 150" in str(result.error)

    def test_user_validation_excessive_age(self) -> None:
        """Test user validation with excessive age."""
        user = User(
            id="test-id",
            name="Valid Name",
            email="valid@example.com",
            age=151,
        )

        result = user.validate_business_rules()
        assert result.is_failure
        assert "Age must be between 0 and 150" in str(result.error)

    def test_user_validation_boundary_age_zero(self) -> None:
        """Test user validation with age 0."""
        user = User(
            id="test-id",
            name="Valid Name",
            email="valid@example.com",
            age=0,
        )

        result = user.validate_business_rules()
        assert result.success

    def test_user_validation_boundary_age_150(self) -> None:
        """Test user validation with age 150."""
        user = User(
            id="test-id",
            name="Valid Name",
            email="valid@example.com",
            age=150,
        )

        result = user.validate_business_rules()
        assert result.success


# =============================================================================
# TEST USER SERVICE
# =============================================================================


class TestUserService:
    """Test UserService implementation."""

    @pytest.fixture
    def user_service(self) -> UserService:
        """Create a fresh UserService instance."""
        return UserService()

    def test_create_user_success(self, user_service: UserService) -> None:
        """Test successful user creation."""
        user_data = UserData(name="Test User", email="test@example.com", age=25)

        result = user_service.create_user(user_data)

        assert result.success
        user = result.unwrap()
        assert user.name == "Test User"
        assert user.email == "test@example.com"
        assert user.age == 25
        assert user.id  # Should have generated ID

    def test_create_user_duplicate(self, user_service: UserService) -> None:
        """Test duplicate user prevention."""
        user_data = UserData(name="Test User", email="test@example.com", age=25)

        # Create first user
        result1 = user_service.create_user(user_data)
        assert result1.success

        # Try to create duplicate
        result2 = user_service.create_user(user_data)
        assert result2.is_failure
        assert "already exists" in str(result2.error)

    def test_create_user_validation_failure(self, user_service: UserService) -> None:
        """Test user creation with validation failure."""
        user_data = UserData(name="A", email="test@example.com", age=25)

        result = user_service.create_user(user_data)

        assert result.is_failure
        assert "Name must be at least 2 characters" in str(result.error)

    def test_find_user_by_email_exists(self, user_service: UserService) -> None:
        """Test finding existing user by email."""
        user_data = UserData(name="Test User", email="test@example.com", age=25)
        user_service.create_user(user_data)

        result = user_service.find_user_by_email("test@example.com")

        assert result.success
        user = result.unwrap()
        assert user is not None
        assert user.email == "test@example.com"

    def test_find_user_by_email_not_exists(self, user_service: UserService) -> None:
        """Test finding non-existent user by email."""
        result = user_service.find_user_by_email("notfound@example.com")

        assert result.success
        assert result.unwrap() is None

    def test_find_user_by_email_invalid(self, user_service: UserService) -> None:
        """Test finding user with invalid email."""
        result = user_service.find_user_by_email("invalid-email")

        assert result.is_failure
        assert "Invalid email format" in str(result.error)


# =============================================================================
# TEST NOTIFICATION SERVICE
# =============================================================================


class TestNotificationService:
    """Test NotificationService implementation."""

    @pytest.fixture
    def notification_service(self) -> NotificationService:
        """Create a NotificationService instance."""
        return NotificationService()

    def test_send_welcome_success(
        self, notification_service: NotificationService,
    ) -> None:
        """Test successful welcome notification."""
        user = User(
            id="test-id",
            name="Test User",
            email="test@example.com",
            age=25,
        )

        # Capture print output
        with contextlib.redirect_stdout(StringIO()) as output:
            result = notification_service.send_welcome(user)

        assert result.success
        assert result.unwrap() is None
        assert "Welcome email sent" in output.getvalue()
        assert "Test User" in output.getvalue()
        assert "test@example.com" in output.getvalue()

    def test_send_welcome_invalid_email(
        self, notification_service: NotificationService,
    ) -> None:
        """Test welcome notification with invalid email."""
        user = User(
            id="test-id",
            name="Test User",
            email="invalid-email",
            age=25,
        )

        result = notification_service.send_welcome(user)

        assert result.is_failure
        assert "Invalid email format" in str(result.error)


# =============================================================================
# TEST USER REGISTRATION SERVICE
# =============================================================================


class TestUserRegistrationService:
    """Test UserRegistrationService orchestration."""

    @pytest.fixture
    def services(self) -> tuple[UserService, NotificationService]:
        """Create service instances."""
        return UserService(), NotificationService()

    @pytest.fixture
    def registration_service(
        self, services: tuple[UserService, NotificationService],
    ) -> UserRegistrationService:
        """Create UserRegistrationService with dependencies."""
        user_service, notification_service = services
        return UserRegistrationService(user_service, notification_service)

    def test_register_user_success(
        self, registration_service: UserRegistrationService,
    ) -> None:
        """Test successful user registration."""
        with contextlib.redirect_stdout(StringIO()) as output:
            result = registration_service.register_user(
                "Test User", "test@example.com", 25,
            )

        assert result.success
        user = result.unwrap()
        assert user.name == "Test User"
        assert user.email == "test@example.com"
        assert "Welcome email sent" in output.getvalue()

    def test_register_user_validation_failure(
        self, registration_service: UserRegistrationService,
    ) -> None:
        """Test registration with validation failure."""
        result = registration_service.register_user("A", "test@example.com", 25)

        assert result.is_failure
        assert "Name must be at least 2 characters" in str(result.error)

    def test_register_user_duplicate(
        self, registration_service: UserRegistrationService,
    ) -> None:
        """Test duplicate user registration."""
        # Register first user
        result1 = registration_service.register_user(
            "Test User", "test@example.com", 25,
        )
        assert result1.success

        # Try to register duplicate
        result2 = registration_service.register_user(
            "Another User", "test@example.com", 30,
        )
        assert result2.is_failure
        assert "already exists" in str(result2.error)

    def test_register_user_notification_failure(
        self, services: tuple[UserService, NotificationService],
    ) -> None:
        """Test registration continues despite notification failure."""
        user_service, _ = services

        # Create a failing notification service
        class FailingNotificationService:
            def send_welcome(self, user: User) -> FlextResult[None]:
                return FlextResult[None].fail("Notification system down")

        registration_service = UserRegistrationService(
            user_service,
            FailingNotificationService(),
        )

        with contextlib.redirect_stdout(StringIO()) as output:
            result = registration_service.register_user(
                "Test User", "test@example.com", 25,
            )

        # Registration should succeed despite notification failure
        assert result.success
        user = result.unwrap()
        assert user.name == "Test User"
        assert "Warning: Welcome notification failed" in output.getvalue()


# =============================================================================
# TEST DEPENDENCY INJECTION SETUP
# =============================================================================


class TestDependencyInjection:
    """Test dependency injection setup and container configuration."""

    def test_setup_container(self) -> None:
        """Test container setup."""
        # Clear any existing registrations
        container = FlextContainer.get_global()
        container.clear()

        result = setup_container()

        assert result.success
        container = result.unwrap()

        # Check services are registered
        assert container.has("user_service")
        assert container.has("notification_service")
        assert container.has("registration_service")

    def test_container_service_retrieval(self) -> None:
        """Test retrieving services from container."""
        # Setup container
        container = FlextContainer.get_global()
        container.clear()
        setup_container()

        # Get user service
        user_service_result = container.get("user_service")
        assert user_service_result.success
        assert isinstance(user_service_result.unwrap(), UserService)

        # Get notification service
        notification_result = container.get("notification_service")
        assert notification_result.success
        assert isinstance(notification_result.unwrap(), NotificationService)

        # Get registration service (factory)
        registration_result = container.get("registration_service")
        assert registration_result.success
        assert isinstance(registration_result.unwrap(), UserRegistrationService)

    def test_factory_registration(self) -> None:
        """Test that factory is registered and can create instances."""
        container = FlextContainer.get_global()
        container.clear()
        setup_container()

        # Get instance from factory
        result = container.get("registration_service")

        assert result.success
        instance = result.unwrap()
        assert isinstance(instance, UserRegistrationService)

        # Test that the created instance works
        registration_service = cast("UserRegistrationService", instance)
        test_result = registration_service.register_user("Test", "test@example.com", 25)
        assert test_result.success


# =============================================================================
# TEST MAIN FUNCTION
# =============================================================================


class TestMainFunction:
    """Test the main demonstration function."""

    def test_main_success(self) -> None:
        """Test successful main function execution."""
        # Clear container
        container = FlextContainer.get_global()
        container.clear()

        # Capture output
        with contextlib.redirect_stdout(StringIO()) as output:
            main()

        output_str = output.getvalue()

        # Check expected outputs
        assert "FlextCore Dependency Injection Demo" in output_str
        assert "Container configured successfully" in output_str
        assert "Registration service retrieved" in output_str

        # Check user registrations
        assert "Registered: Alice Johnson" in output_str
        assert "Registered: Bob Smith" in output_str
        assert "Registered: Charlie Brown" in output_str

        # Check duplicate prevention
        assert "Duplicate prevented" in output_str

        # Check user lookup
        assert "Found user: Bob Smith" in output_str

        # Check validations
        assert "Empty name validation" in output_str
        assert "Invalid email validation" in output_str
        assert "Invalid age validation" in output_str

        # Check completion
        assert "Dependency injection demo completed successfully" in output_str

    def test_main_container_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main function with container setup failure."""

        # Mock setup_container to fail
        def failing_setup() -> FlextResult[FlextContainer]:
            return FlextResult[FlextContainer].fail("Container setup failed")

        monkeypatch.setattr(di_example, "setup_container", failing_setup)

        with contextlib.redirect_stdout(StringIO()) as output:
            main()

        output_str = output.getvalue()
        assert "Container setup failed" in output_str
        assert "Dependency injection demo completed" not in output_str

    def test_main_service_retrieval_failure(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test main function with service retrieval failure."""

        # Setup container but don't register registration service
        def partial_setup() -> FlextResult[FlextContainer]:
            container = FlextContainer.get_global()
            container.clear()
            container.register("user_service", UserService())
            container.register("notification_service", NotificationService())
            # Don't register registration_service
            return FlextResult[FlextContainer].ok(container)

        monkeypatch.setattr(di_example, "setup_container", partial_setup)

        with contextlib.redirect_stdout(StringIO()) as output:
            main()

        output_str = output.getvalue()
        assert "Failed to get registration service" in output_str
        assert "Dependency injection demo completed" not in output_str


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self) -> None:
        """Test complete workflow from setup to registration."""
        # Setup
        container = FlextContainer.get_global()
        container.clear()
        setup_container()

        # Get registration service
        registration_result = container.get("registration_service")
        assert registration_result.success
        registration_service = cast(
            "UserRegistrationService", registration_result.unwrap(),
        )

        # Register multiple users
        users = [
            ("User One", "user1@test.com", 20),
            ("User Two", "user2@test.com", 30),
            ("User Three", "user3@test.com", 40),
        ]

        registered_users = []
        for name, email, age in users:
            result = registration_service.register_user(name, email, age)
            assert result.success
            registered_users.append(result.unwrap())

        # Verify all users were created
        user_service_result = container.get("user_service")
        assert user_service_result.success
        user_service = cast("UserService", user_service_result.unwrap())

        for user in registered_users:
            lookup_result = user_service.find_user_by_email(user.email)
            assert lookup_result.success
            found_user = lookup_result.unwrap()
            assert found_user is not None
            assert found_user.email == user.email

    def test_error_propagation(self) -> None:
        """Test that errors propagate correctly through the system."""
        container = FlextContainer.get_global()
        container.clear()
        setup_container()

        registration_result = container.get("registration_service")
        assert registration_result.success
        registration_service = cast(
            "UserRegistrationService", registration_result.unwrap(),
        )

        # Test various validation errors
        test_cases = [
            ("", "test@test.com", 25, "at least 2 characters"),
            ("Valid", "invalid", 25, "Invalid email"),
            ("Valid", "test@test.com", -1, "between 0 and 150"),
        ]

        for name, email, age, expected_error in test_cases:
            result = registration_service.register_user(name, email, age)
            assert result.is_failure
            assert expected_error in str(result.error)


# =============================================================================
# TEST RAILWAY PATTERN INTEGRATION
# =============================================================================


class TestRailwayPatternIntegration:
    """Test integration of railway patterns with flext-core components."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        container = FlextContainer.get_global()
        container.clear()
        setup_container()

    def test_flext_result_chain_operations(self) -> None:
        """Test chaining FlextResult operations."""
        user_service = UserService()

        # Create a valid user
        user_data = UserData(name="Chain Test", email="chain@example.com", age=28)

        result = user_service.create_user(user_data)
        assert result.is_success

        # Test that we can access user properties
        user = result.unwrap()
        assert user.id
        assert user.email == "chain@example.com"

    def test_railway_pattern_error_propagation(self) -> None:
        """Test that errors properly propagate through railway pattern."""
        registration_service = UserRegistrationService(
            UserService(), NotificationService(),
        )

        # Test with invalid request
        result = registration_service.register_user("", "invalid", -10)
        assert result.is_failure

        # Error should be from validation
        assert "at least 2 characters" in str(result.error)

    def test_railway_pattern_success_propagation(self) -> None:
        """Test that success properly propagates through railway pattern."""
        registration_service = UserRegistrationService(
            UserService(), NotificationService(),
        )

        # Test with valid request
        result = registration_service.register_user(
            "Success Test", "success@example.com", 30,
        )
        assert result.is_success

        # Should have proper data structure
        user = result.unwrap()
        assert isinstance(user, User)
        assert user.name == "Success Test"
        assert user.email == "success@example.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
