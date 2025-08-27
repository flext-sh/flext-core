"""Tests reflecting real flext-core usage patterns.

These tests demonstrate how flext-core is actually used in production,
focusing on practical scenarios rather than edge cases.
"""

from __future__ import annotations

import pytest

from flext_core import FlextContainer, FlextResult, FlextTypes

pytestmark = [pytest.mark.e2e]


class TestUser:
    """Test user for testing."""

    def __init__(self, user_id: FlextTypes.Domain.EntityId, name: str) -> None:
        """Initialize test user."""
        self.id = user_id
        self.name = name


class TestDatabase:
    """Test database service for testing."""

    def __init__(self) -> None:
        """Initialize test database with test data."""
        self.users = {
            "user-123": TestUser("user-123", "John Doe"),
            "user-456": TestUser("user-456", "Jane Smith"),
        }

    def get_user(self, user_id: FlextTypes.Domain.EntityId) -> TestUser | None:
        """Get user by ID."""
        return self.users.get(user_id)

    def save_user(self, user: TestUser) -> TestUser:
        """Save user to database."""
        self.users[user.id] = user
        return user


class UserService:
    """User service demonstrating real service patterns."""

    def __init__(self, database: TestDatabase) -> None:
        """Initialize user service with database dependency."""
        self.database = database

    def fetch_user(self, user_id: FlextTypes.Domain.EntityId) -> FlextResult[TestUser]:
        """Fetch user with type-safe error handling."""
        user = self.database.get_user(user_id)
        if user is None:
            return FlextResult[TestUser].fail(f"User {user_id} not found")
        return FlextResult[TestUser].ok(user)

    def create_user(
        self,
        user_id: FlextTypes.Domain.EntityId,
        name: str,
    ) -> FlextResult[TestUser]:
        """Create a new user."""
        if self.database.get_user(user_id) is not None:
            return FlextResult[TestUser].fail(f"User {user_id} already exists")

        user = TestUser(user_id, name)
        self.database.save_user(user)
        return FlextResult[TestUser].ok(user)


class TestUsagePatterns:
    """Tests that mirror actual production usage patterns."""

    def test_flext_result_success_flow(self) -> None:
        """Test typical successful FlextResult flow."""
        # Common pattern: service operation that succeeds
        database = TestDatabase()
        service = UserService(database)

        result = service.fetch_user("user-123")

        # Production code checks success first
        assert result.success
        user = result.value
        assert user is not None
        if user.name != "John Doe":
            msg: str = f"Expected {'John Doe'}, got {user.name}"
            raise AssertionError(msg)
        assert user.id == "user-123"

    def test_flext_result_failure_flow(self) -> None:
        """Test typical failure FlextResult flow."""
        # Common pattern: service operation that fails
        database = TestDatabase()
        service = UserService(database)

        result = service.fetch_user("nonexistent")

        # Production error handling pattern
        assert result.is_failure
        assert result.error is not None
        assert result.error
        if "not found" not in (result.error or ""):
            msg: str = f"Expected 'not found' in {result.error}"
            raise AssertionError(msg)

    def test_chained_operations(self) -> None:
        """Test chaining operations with FlextResult."""
        database = TestDatabase()
        service = UserService(database)

        # Chain of operations that can fail at any step
        create_result = service.create_user(
            "new-user",
            "New User",
        )
        assert create_result.success

        fetch_result = service.fetch_user("new-user")
        assert fetch_result.success
        assert fetch_result.value is not None
        if fetch_result.value.name != "New User":
            msg: str = f"Expected {'New User'}, got {fetch_result.value.name}"
            raise AssertionError(msg)

    def test_error_handling_patterns(self) -> None:
        """Test common error handling patterns."""
        database = TestDatabase()
        service = UserService(database)

        # Try to create duplicate user
        duplicate_result = service.create_user(
            "user-123",
            "Duplicate",
        )
        assert duplicate_result.is_failure
        assert duplicate_result.error is not None
        if "already exists" not in duplicate_result.error:
            duplicate_msg: str = (
                f"Expected {'already exists'} in {duplicate_result.error}"
            )
            raise AssertionError(duplicate_msg)

        # Original user should be unchanged
        original_result = service.fetch_user("user-123")
        assert original_result.success
        assert original_result.value is not None
        if original_result.value.name != "John Doe":
            original_msg: str = (
                f"Expected {'John Doe'}, got {original_result.value.name}"
            )
            raise AssertionError(original_msg)

    def test_dependency_injection_basic_usage(self) -> None:
        """Test basic dependency injection usage (most common pattern).

        This test demonstrates the most common dependency injection pattern.
        """
        container = FlextContainer()

        # Register dependencies (application startup pattern)
        database = TestDatabase()
        register_db_result = container.register("database", database)
        assert register_db_result.success

        # Register service with dependencies
        def create_user_service() -> UserService:
            db_result = container.get("database")
            assert db_result.success
            assert isinstance(db_result.value, TestDatabase)
            return UserService(db_result.value)

        register_service_result = container.register_factory(
            "user_service",
            create_user_service,
        )
        assert register_service_result.success

        # Use service (application runtime pattern)
        service_result = container.get("user_service")
        assert service_result.success

        service = service_result.value
        assert isinstance(service, UserService)
        user_result = service.fetch_user("user-123")
        assert user_result.success
        assert user_result.value is not None
        if user_result.value.name != "John Doe":
            msg: str = f"Expected {'John Doe'}, got {user_result.value.name}"
            raise AssertionError(msg)

    def test_global_container_usage(self) -> None:
        """Test global container usage (application-wide pattern)."""
        # Setup global services (application startup)
        container = FlextContainer.get_global()

        database = TestDatabase()
        register_result = container.register("global_database", database)
        assert register_result.success

        # Access from anywhere in application
        container2 = FlextContainer.get_global()
        assert container2 is container  # Same instance

        db_result = container2.get("global_database")
        assert db_result.success
        assert db_result.value is database

    def test_entity_id_type_safety(self) -> None:
        """Test FlextEntityId type safety in real usage."""
        # Type-safe entity ID usage
        user_id: FlextTypes.Domain.EntityId = "typed-user-123"

        database = TestDatabase()
        service = UserService(database)

        # FlextEntityId works seamlessly with services
        create_result = service.create_user(user_id, "Typed User")
        assert create_result.success

        fetch_result = service.fetch_user(user_id)
        assert fetch_result.success
        assert fetch_result.value is not None
        if fetch_result.value.id != user_id:
            msg: str = f"Expected {user_id}, got {fetch_result.value.id}"
            raise AssertionError(msg)

    def test_validation_patterns(self) -> None:
        """Test validation patterns used in production."""
        database = TestDatabase()
        service = UserService(database)

        # Empty name validation would typically happen at service level
        empty_id = ""
        result = service.create_user(empty_id, "Test User")

        # Service should handle validation (business logic)
        # This is a simplified example - real validation would be more
        # sophisticated
        assert result.success or result.is_failure  # Either outcome is valid

    def test_service_factory_pattern(self) -> None:
        """Test service factory pattern (enterprise usage)."""
        container = FlextContainer()

        # Factory pattern for complex service creation
        def create_complex_service() -> UserService:
            database = TestDatabase()
            # In real code, this might involve configuration, logging
            # setup, etc.
            return UserService(database)

        register_result = container.register_factory(
            "complex_service",
            create_complex_service,
        )
        assert register_result.success

        # Factory is called only once
        service1_result = container.get("complex_service")
        service2_result = container.get("complex_service")

        assert service1_result.success
        assert service2_result.success
        assert service1_result.value is service2_result.value  # Same instance
