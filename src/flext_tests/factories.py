"""Advanced test factories using factory_boy for consistent test data generation.

Provides factory_boy-based factories for creating test objects with proper
relationships, sequences, and customization following SOLID principles.
"""
# mypy: disable-error-code="var-annotated,valid-type,no-untyped-call,name-defined,misc,no-any-return"

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar, cast

import factory
from factory.declarations import LazyAttribute, LazyFunction, Sequence
from factory.faker import Faker

from flext_core import (
    FlextConstants,
    FlextModels,
    FlextResult,
)

AuditLogValue = str | int | bool | None | dict[str, str | int | bool | None] | list[str]
AuditLog = dict[str, AuditLogValue]


class RepositoryError(Exception):
    """Custom exception for repository operations."""


# Base models for testing (these would typically come from domain models)
class TestUser(FlextModels.Config):
    """Test user model for factory testing."""

    id: str
    name: str
    email: str
    age: int
    is_active: bool
    created_at: datetime
    metadata: dict[str, object]


class TestConfig(FlextModels.Config):
    """Test configuration model for factory testing."""

    database_url: str
    log_level: str
    debug: bool
    timeout: int
    max_connections: int
    features: list[str]


class TestField(FlextModels.Config):
    """Test field model for factory testing."""

    field_id: str
    field_name: str
    field_type: str
    required: bool
    description: str
    min_length: int | None = None
    max_length: int | None = None
    min_value: int | None = None
    max_value: int | None = None
    default_value: object = None
    pattern: str | None = None


class BaseTestEntity(FlextModels.Config):
    """Base test entity for domain testing."""

    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    version: int = 1
    metadata: ClassVar[dict[str, object]] = {}


class BaseTestValueObject(FlextModels.Config):
    """Base test value object for domain testing."""

    value: str
    description: str
    category: str
    tags: ClassVar[list[str]] = []


# Factory Boy Factories
class UserFactory(factory.Factory[TestUser]):
    """Factory for creating test users with factory_boy."""

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Factory meta compatibility."""

        model = TestUser

    id = LazyAttribute(lambda _: str(uuid.uuid4()))
    name = Faker("name")
    email = Faker("email")
    age = Faker("random_int", min=18, max=80)
    is_active: bool = True
    created_at = LazyAttribute(lambda _: datetime.now(UTC))
    metadata = LazyFunction(
        lambda: {"department": "engineering", "level": "senior", "team": "backend"}
    )


class AdminUserFactory(UserFactory):
    """Factory for admin users."""

    # Use build/create methods to override values instead of class attributes
    @classmethod
    def _adjust_kwargs(cls, **kwargs: object) -> dict[str, object]:
        # Override default kwargs for admin users
        if "name" not in kwargs:
            kwargs["name"] = "Admin User"
        if "email" not in kwargs:
            kwargs["email"] = "admin@example.com"
        return cast("dict[str, object]", super()._adjust_kwargs(**kwargs))

    metadata = LazyFunction(
        lambda: {"department": "admin", "level": "admin", "permissions": "all"},
    )


class InactiveUserFactory(UserFactory):
    """Factory for inactive users."""

    is_active = False
    metadata = LazyFunction(
        lambda: {
            "department": "archived",
            "level": "inactive",
            "archived_at": str(datetime.now(UTC)),
        },
    )


class ConfigFactory(factory.Factory[TestConfig]):
    """Factory for creating test configurations."""

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Factory meta compatibility."""

        model = TestConfig

    database_url: str = "postgresql://test:test@localhost/test_db"
    log_level: str = "DEBUG"
    debug: bool = True
    timeout: int = 30
    max_connections: int = 100
    features = LazyFunction(lambda: ["auth", "cache", "metrics", "monitoring"])


class ProductionConfigFactory(ConfigFactory):
    """Factory for production-like configurations."""

    database_url = "postgresql://prod:prod@prod-db/prod_db"
    log_level = "INFO"
    debug = False
    timeout = 60
    max_connections = 500
    features = LazyFunction(
        lambda: ["auth", "cache", "metrics", "monitoring", "alerts"],
    )


class StringFieldFactory(factory.Factory[TestField]):
    """Factory for string field testing."""

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Factory meta compatibility."""

        model = TestField

    field_id = LazyAttribute(lambda _: str(uuid.uuid4()))
    field_name = Sequence(lambda n: f"string_field_{n}")
    field_type: str = FlextConstants.Enums.FieldType.STRING.value
    required: bool = True
    description = LazyAttribute(
        lambda obj: f"Test string field: {getattr(obj, 'field_name', 'unknown')}",
    )
    min_length = 1
    max_length = 100
    pattern = r"^[a-zA-Z0-9_]+$"


class IntegerFieldFactory(factory.Factory[TestField]):
    """Factory for integer field testing."""

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Factory meta compatibility."""

        model = TestField

    field_id = LazyAttribute(lambda _: str(uuid.uuid4()))
    field_name = Sequence(lambda n: f"integer_field_{n}")
    field_type: str = FlextConstants.Enums.FieldType.INTEGER.value
    required: bool = True
    description = LazyAttribute(
        lambda obj: f"Test integer field: {getattr(obj, 'field_name', 'unknown')}",
    )
    min_value = 0
    max_value = 1000


class BooleanFieldFactory(factory.Factory[TestField]):
    """Factory for boolean field testing."""

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Factory meta compatibility."""

        model = TestField

    field_id = LazyAttribute(lambda _: str(uuid.uuid4()))
    field_name = Sequence(lambda n: f"boolean_field_{n}")
    field_type = FlextConstants.Enums.FieldType.BOOLEAN.value
    required = True
    description = LazyAttribute(
        lambda obj: f"Test boolean field: {getattr(obj, 'field_name', 'unknown')}",
    )
    default_value = False


class FloatFieldFactory(factory.Factory[TestField]):
    """Factory for float field testing."""

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Factory meta compatibility."""

        model = TestField

    field_id = LazyAttribute(lambda _: str(uuid.uuid4()))
    field_name = Sequence(lambda n: f"float_field_{n}")
    field_type = FlextConstants.Enums.FieldType.FLOAT.value
    required = True
    description = LazyAttribute(
        lambda obj: f"Test float field: {getattr(obj, 'field_name', 'unknown')}",
    )
    min_value = 0.0
    max_value = 1000.0


class TestEntityFactory(factory.Factory[BaseTestEntity]):
    """Factory for creating test entities."""

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Factory meta compatibility."""

        model = BaseTestEntity

    id = LazyAttribute(lambda _: str(uuid.uuid4()))
    name = Faker("company")
    created_at = LazyAttribute(lambda _: datetime.now(UTC))
    updated_at = LazyAttribute(lambda _: datetime.now(UTC))
    version = 1
    # metadata é ClassVar, não é passado na inicialização


class TestValueObjectFactory(factory.Factory[BaseTestValueObject]):
    """Factory for creating test value objects."""

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Factory meta compatibility."""

        model = BaseTestValueObject

    value = Faker("word")
    description = Faker("sentence")
    category = Faker("word")
    # tags é ClassVar, não é passado na inicialização


# FlextResult factories
class FlextResultFactory:
    """Factory for creating FlextResult instances for testing."""

    @staticmethod
    def success(data: object = None) -> FlextResult[object]:
        """Create successful FlextResult."""
        return FlextResult[object].ok(data or "test_success_data")

    @staticmethod
    def failure(
        error: str = "test_error",
        error_code: str = "TEST_ERROR",
    ) -> FlextResult[object]:
        """Create failed FlextResult."""
        return FlextResult[object].fail(error, error_code=error_code)

    @staticmethod
    def success_with_user() -> FlextResult[TestUser]:
        """Create successful FlextResult with user data."""
        user = cast("TestUser", UserFactory())
        return FlextResult[TestUser].ok(user)

    @staticmethod
    def success_with_config() -> FlextResult[TestConfig]:
        """Create successful FlextResult with config data."""
        config = cast("TestConfig", ConfigFactory())
        return FlextResult[TestConfig].ok(config)

    @staticmethod
    def create_success(data: object = None) -> FlextResult[object]:
        """Create successful FlextResult (alias for success method)."""
        return FlextResultFactory.success(data)

    @staticmethod
    def create_failure(
        error: str = "test_error",
        error_code: str = "TEST_ERROR",
    ) -> FlextResult[object]:
        """Create failed FlextResult (alias for failure method)."""
        return FlextResultFactory.failure(error, error_code)

    @staticmethod
    def create_batch(size: int = 10) -> list[FlextResult[object]]:
        """Create batch of successful FlextResults."""
        return [FlextResultFactory.success(f"test_data_{i}") for i in range(size)]


# Sequence generators for common patterns
class SequenceGenerators:
    """Generators for common sequence patterns in tests."""

    @staticmethod
    def entity_id_sequence() -> str:
        """Generate sequence of entity IDs."""
        return f"test_entity_{uuid.uuid4()}"

    @staticmethod
    def timestamp_sequence() -> datetime:
        """Generate sequence of timestamps."""
        return datetime.now(UTC)

    @staticmethod
    def email_sequence(domain: str = "example.com") -> str:
        """Generate sequence of email addresses."""
        return f"test_{uuid.uuid4().hex[:8]}@{domain}"

    @staticmethod
    def username_sequence() -> str:
        """Generate sequence of usernames."""
        return f"user_{uuid.uuid4().hex[:8]}"


# Batch factory functions for performance testing
class BatchFactories:
    """Batch creation utilities for performance and integration testing."""

    @staticmethod
    def create_users(count: int = 10) -> list[TestUser]:
        """Create batch of test users."""
        return UserFactory.create_batch(count)

    @staticmethod
    def create_mixed_users(count: int = 10) -> list[TestUser]:
        """Create batch of mixed user types."""
        users: list[TestUser] = []
        for i in range(count):
            if i % 3 == 0:
                users.append(cast("TestUser", AdminUserFactory()))
            elif i % 5 == 0:
                users.append(cast("TestUser", InactiveUserFactory()))
            else:
                users.append(cast("TestUser", UserFactory()))
        return users

    @staticmethod
    def create_field_matrix() -> list[TestField]:
        """Create comprehensive field test matrix."""
        fields: list[TestField] = []
        fields.extend(StringFieldFactory.create_batch(3))
        fields.extend(IntegerFieldFactory.create_batch(3))
        fields.extend(BooleanFieldFactory.create_batch(2))
        fields.extend(FloatFieldFactory.create_batch(2))
        return fields


# Edge case data generators
class EdgeCaseGenerators:
    """Generators for edge case testing scenarios."""

    @staticmethod
    def unicode_strings() -> list[str]:
        """Generate unicode test strings."""
        return ["🚀", "测试", "مرحبا", "🔥🎯", "Ñoño", "café"]

    @staticmethod
    def special_characters() -> list[str]:
        """Generate special character test strings."""
        return ["!@#$%^&*()", "\n\t\r", "\\", "'\"", "<script>", "SELECT * FROM"]

    @staticmethod
    def boundary_numbers() -> list[int | float]:
        """Generate boundary number test values."""
        return [0, -1, 1, 999999999, -999999999, 1e-10, float("inf"), float("-inf")]

    @staticmethod
    def empty_values() -> list[object]:
        """Generate empty/null test values."""
        return ["", [], {}, None, 0, False]

    @staticmethod
    def large_values() -> list[object]:
        """Generate large value test cases."""
        return [
            "x" * 10000,
            list(range(1000)),
            {f"key_{i}": f"value_{i}" for i in range(100)},
        ]


# Utility functions for common test patterns
def create_test_hierarchy() -> dict[str, object]:
    """Create hierarchical test data structure."""
    return {
        "root": cast("TestUser", UserFactory()),
        "children": UserFactory.create_batch(3),  # Remove redundant cast
        "admin": cast("TestUser", AdminUserFactory()),
        "config": cast("TestConfig", ConfigFactory()),
        "fields": BatchFactories.create_field_matrix(),
    }


def create_validation_test_cases() -> list[dict[str, object]]:
    """Create comprehensive validation test cases."""
    return [
        {
            "name": "valid_user",
            "data": cast("TestUser", UserFactory()),
            "expected_valid": True,
        },
        {
            "name": "invalid_email",
            "data": TestUser(
                id="test-id",
                name="Test User",
                email="invalid-email",
                age=25,
                is_active=True,
                created_at=datetime.now(UTC),
                metadata={},
            ),
            "expected_valid": False,
        },
        {
            "name": "negative_age",
            "data": TestUser(
                id="test-id",
                name="Test User",
                email="test@example.com",
                age=-5,
                is_active=True,
                created_at=datetime.now(UTC),
                metadata={},
            ),
            "expected_valid": False,
        },
        {
            "name": "unicode_name",
            "data": TestUser(
                id="test-id",
                name="测试用户",
                email="test@example.com",
                age=25,
                is_active=True,
                created_at=datetime.now(UTC),
                metadata={},
            ),
            "expected_valid": True,
        },
    ]


# Convenience functions for backward compatibility
def success_result(data: object = "test_data") -> FlextResult[object]:
    """Create successful FlextResult."""
    return FlextResult[object].ok(data)


def failure_result(
    error: str = "Test error",
    error_code: str = "TEST_ERROR",
) -> FlextResult[object]:
    """Create failed FlextResult."""
    return FlextResult[object].fail(error, error_code=error_code)


def validation_failure(field: str = "test_field") -> FlextResult[object]:
    """Create validation failure FlextResult."""
    return FlextResult[object].fail(
        f"Validation failed for field: {field}",
        error_code="VALIDATION_ERROR",
    )


# Export commonly used factories and utilities
__all__ = [
    "AdminUserFactory",
    "BaseTestEntity",
    "BaseTestValueObject",
    "BatchFactories",
    "BooleanFieldFactory",
    "ConfigFactory",
    "EdgeCaseGenerators",
    "FailingUserRepository",
    "FlextResultFactory",
    "FloatFieldFactory",
    "InMemoryUserRepository",
    "InactiveUserFactory",
    "IntegerFieldFactory",
    "ProductionConfigFactory",
    "RealAuditService",
    "RealEmailService",
    "RepositoryError",
    "SequenceGenerators",
    "StringFieldFactory",
    "TestConfig",
    "TestEntityFactory",
    "TestField",
    "TestUser",
    "TestValueObjectFactory",
    "User",
    "UserFactory",
    "create_test_hierarchy",
    "create_validation_test_cases",
    "failure_result",
    "success_result",
    "validation_failure",
]
"""Real implementations for testing - eliminating mocks.

Provides real implementations of repositories, services, and other dependencies
to replace mock objects in tests for true functional testing.
"""


@dataclass
class User:
    """Real user model for testing."""

    id: str
    username: str
    email: str
    age: int = 18
    is_admin: bool = False
    created_at: datetime | None = None


class InMemoryUserRepository:
    """Real in-memory user repository implementation."""

    def __init__(self) -> None:
        self._users: dict[str, User] = {}
        self._users_by_username: dict[str, User] = {}

    def create(self, user: User) -> bool:
        """Create a user."""
        if user.id in self._users or user.username in self._users_by_username:
            return False

        if user.created_at is None:
            user.created_at = datetime.now(UTC)

        self._users[user.id] = user
        self._users_by_username[user.username] = user
        return True

    def find_by_username(
        self,
        username: str,
    ) -> dict[str, str | int | bool | None] | None:
        """Find user by username."""
        user = self._users_by_username.get(username)
        if user is None:
            return None

        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "age": user.age,
            "is_admin": user.is_admin,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }

    def find_by_id(self, user_id: str) -> dict[str, str | int | bool | None] | None:
        """Find user by ID."""
        user = self._users.get(user_id)
        if user is None:
            return None

        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "age": user.age,
            "is_admin": user.is_admin,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }

    def update(self, user_id: str, updates: dict[str, str | int | bool | None]) -> bool:
        """Update a user."""
        if user_id not in self._users:
            return False

        user = self._users[user_id]
        old_username = user.username

        # Update user attributes
        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)

        # Update username index if changed
        if old_username != user.username:
            del self._users_by_username[old_username]
            self._users_by_username[user.username] = user

        return True

    def delete(self, user_id: str) -> bool:
        """Delete a user."""
        if user_id not in self._users:
            return False

        user = self._users[user_id]
        del self._users[user_id]
        del self._users_by_username[user.username]
        return True

    def count(self) -> int:
        """Get user count."""
        return len(self._users)

    def clear(self) -> None:
        """Clear all users."""
        self._users.clear()
        self._users_by_username.clear()


class RealEmailService:
    """Real email service implementation for testing."""

    def __init__(self) -> None:
        self.sent_emails: list[dict[str, str | int | bool | None]] = []
        self.should_fail = False

    def send_welcome_email(
        self,
        user: dict[str, str | int | bool | None],
    ) -> FlextResult[bool]:
        """Send welcome email."""
        if self.should_fail:
            return FlextResult[bool].fail("Email service unavailable")

        email_data = {
            "to": user["email"],
            "subject": "Welcome!",
            "body": f"Welcome {user['username']}!",
            "sent_at": datetime.now(UTC).isoformat(),
        }

        self.sent_emails.append(email_data)
        return FlextResult[bool].ok(data=True)

    def get_sent_emails(self) -> list[dict[str, str | int | bool | None]]:
        """Get all sent emails."""
        return self.sent_emails.copy()

    def clear_sent_emails(self) -> None:
        """Clear sent emails."""
        self.sent_emails.clear()


class RealAuditService:
    """Real audit service implementation for testing."""

    def __init__(self) -> None:
        self.audit_logs: list[AuditLog] = []

    def log_user_created(self, user: dict[str, str | int | bool | None]) -> None:
        """Log user creation event."""
        audit_entry: AuditLog = {
            "event": "USER_CREATED",
            "user_id": user["id"],
            "username": user["username"],
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": {
                "email": user["email"],
                "is_admin": user.get("is_admin", False),
            },
        }

        self.audit_logs.append(audit_entry)

    def log_user_updated(
        self,
        user_id: str,
        changes: dict[str, str | int | bool | None],
    ) -> None:
        """Log user update event."""
        audit_entry: AuditLog = {
            "event": "USER_UPDATED",
            "user_id": user_id,
            "changes": changes,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self.audit_logs.append(audit_entry)

    def get_audit_logs(self) -> list[AuditLog]:
        """Get all audit logs."""
        return self.audit_logs.copy()

    def clear_audit_logs(self) -> None:
        """Clear audit logs."""
        self.audit_logs.clear()


class FailingUserRepository(InMemoryUserRepository):
    """Repository implementation that simulates database failures."""

    def __init__(
        self,
        *,
        fail_on_create: bool = True,
        error_message: str = "Database connection failed",
    ) -> None:
        self._fail_on_create = fail_on_create
        self._error_message = error_message
        self._users: dict[str, User] = {}
        self._users_by_username: dict[str, User] = {}

    def find_by_username(
        self,
        username: str,
    ) -> dict[str, str | int | bool | None] | None:
        """Find user by username."""
        user = self._users_by_username.get(username)
        if user is None:
            return None

        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "age": user.age,
            "is_admin": user.is_admin,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }

    def create(self, user: User) -> bool:
        """Create user - fails if configured to fail."""
        if self._fail_on_create:
            raise RepositoryError(self._error_message)

        if user.id in self._users or user.username in self._users_by_username:
            return False
        self._users[user.id] = user
        self._users_by_username[user.username] = user
        return True
