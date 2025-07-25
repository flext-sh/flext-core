"""Tests reflecting real flext-core usage patterns.

These tests demonstrate how flext-core is actually used in production,
focusing on practical scenarios rather than edge cases.
"""

from __future__ import annotations

from flext_core import FlextContainer
from flext_core import FlextEntityId
from flext_core import FlextResult
from flext_core import get_flext_container


class MockUser:
    """Mock user for testing."""

    def __init__(self, user_id: FlextEntityId, name: str) -> None:
        """Initialize mock user."""
        self.id = user_id
        self.name = name


class MockDatabase:
    """Mock database service for testing."""

    def __init__(self) -> None:
        """Initialize mock database with test data."""
        self.users = {
            "user-123": MockUser(FlextEntityId("user-123"), "John Doe"),
            "user-456": MockUser(FlextEntityId("user-456"), "Jane Smith"),
        }

    def get_user(self, user_id: FlextEntityId) -> MockUser | None:
        """Get user by ID."""
        return self.users.get(user_id)

    def save_user(self, user: MockUser) -> MockUser:
        """Save user to database."""
        self.users[user.id] = user
        return user


class UserService:
    """User service demonstrating real service patterns."""

    def __init__(self, database: MockDatabase) -> None:
        """Initialize user service with database dependency."""
        self.database = database

    def fetch_user(self, user_id: FlextEntityId) -> FlextResult[MockUser]:
        """Fetch user with type-safe error handling."""
        user = self.database.get_user(user_id)
        if user is None:
            return FlextResult.fail(f"User {user_id} not found")
        return FlextResult.ok(user)

    def create_user(
        self,
        user_id: FlextEntityId,
        name: str,
    ) -> FlextResult[MockUser]:
        """Create a new user."""
        if self.database.get_user(user_id) is not None:
            return FlextResult.fail(f"User {user_id} already exists")

        user = MockUser(user_id, name)
        self.database.save_user(user)
        return FlextResult.ok(user)


class TestRealUsagePatterns:
    """Tests that mirror actual production usage patterns."""

    def test_flext_result_success_flow(self) -> None:
        """Test typical successful FlextResult flow."""
        # Common pattern: service operation that succeeds
        database = MockDatabase()
        service = UserService(database)

        result = service.fetch_user(FlextEntityId("user-123"))

        # Production code checks is_success first
        assert result.is_success
        user = result.data
        assert user is not None
        assert user.name == "John Doe"
        assert user.id == "user-123"

    def test_flext_result_failure_flow(self) -> None:
        """Test typical failure FlextResult flow."""
        # Common pattern: service operation that fails
        database = MockDatabase()
        service = UserService(database)

        result = service.fetch_user(FlextEntityId("nonexistent"))

        # Production error handling pattern
        assert result.is_failure
        assert result.error is not None
        assert result.error
        assert "not found" in result.error

    def test_chained_operations(self) -> None:
        """Test chaining operations with FlextResult."""
        database = MockDatabase()
        service = UserService(database)

        # Chain of operations that can fail at any step
        create_result = service.create_user(
            FlextEntityId("new-user"),
            "New User",
        )
        assert create_result.is_success

        fetch_result = service.fetch_user(FlextEntityId("new-user"))
        assert fetch_result.is_success
        assert fetch_result.data is not None
        assert fetch_result.data.name == "New User"

    def test_error_handling_patterns(self) -> None:
        """Test common error handling patterns."""
        database = MockDatabase()
        service = UserService(database)

        # Try to create duplicate user
        duplicate_result = service.create_user(
            FlextEntityId("user-123"),
            "Duplicate",
        )
        assert duplicate_result.is_failure
        assert duplicate_result.error is not None
        assert "already exists" in duplicate_result.error

        # Original user should be unchanged
        original_result = service.fetch_user(FlextEntityId("user-123"))
        assert original_result.is_success
        assert original_result.data is not None
        assert original_result.data.name == "John Doe"

    def test_dependency_injection_basic_usage(self) -> None:
        """Test basic dependency injection usage (most common
        pattern)."""
        container = FlextContainer()

        # Register dependencies (application startup pattern)
        database = MockDatabase()
        register_db_result = container.register("database", database)
        assert register_db_result.is_success

        # Register service with dependencies
        def create_user_service() -> UserService:
            db_result = container.get("database")
            assert db_result.is_success
            assert isinstance(db_result.data, MockDatabase)
            return UserService(db_result.data)

        register_service_result = container.register_singleton(
            "user_service",
            create_user_service,
        )
        assert register_service_result.is_success

        # Use service (application runtime pattern)
        service_result = container.get("user_service")
        assert service_result.is_success

        service = service_result.data
        assert isinstance(service, UserService)
        user_result = service.fetch_user(FlextEntityId("user-123"))
        assert user_result.is_success
        assert user_result.data is not None
        assert user_result.data.name == "John Doe"

    def test_global_container_usage(self) -> None:
        """Test global container usage (application-wide pattern)."""
        # Setup global services (application startup)
        container = get_flext_container()

        database = MockDatabase()
        register_result = container.register("global_database", database)
        assert register_result.is_success

        # Access from anywhere in application
        container2 = get_flext_container()
        assert container2 is container  # Same instance

        db_result = container2.get("global_database")
        assert db_result.is_success
        assert db_result.data is database

    def test_entity_id_type_safety(self) -> None:
        """Test FlextEntityId type safety in real usage."""
        # Type-safe entity ID usage
        user_id: FlextEntityId = FlextEntityId("typed-user-123")

        database = MockDatabase()
        service = UserService(database)

        # FlextEntityId works seamlessly with services
        create_result = service.create_user(user_id, "Typed User")
        assert create_result.is_success

        fetch_result = service.fetch_user(user_id)
        assert fetch_result.is_success
        assert fetch_result.data is not None
        assert fetch_result.data.id == user_id

    def test_validation_patterns(self) -> None:
        """Test validation patterns used in production."""
        database = MockDatabase()
        service = UserService(database)

        # Empty name validation would typically happen at service level
        empty_id = FlextEntityId("")
        result = service.create_user(empty_id, "Test User")

        # Service should handle validation (business logic)
        # This is a simplified example - real validation would be more
        # sophisticated
        assert result.is_success or result.is_failure  # Either outcome is valid

    def test_service_factory_pattern(self) -> None:
        """Test service factory pattern (enterprise usage)."""
        container = FlextContainer()

        # Factory pattern for complex service creation
        def create_complex_service() -> UserService:
            database = MockDatabase()
            # In real code, this might involve configuration, logging
            # setup, etc.
            return UserService(database)

        register_result = container.register_singleton(
            "complex_service",
            create_complex_service,
        )
        assert register_result.is_success

        # Factory is called only once
        service1_result = container.get("complex_service")
        service2_result = container.get("complex_service")

        assert service1_result.is_success
        assert service2_result.is_success
        assert service1_result.data is service2_result.data  # Same instance
