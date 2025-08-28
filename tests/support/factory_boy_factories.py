"""Advanced test factories using factory_boy for consistent test data generation.

Provides factory_boy-based factories for creating test objects with proper
relationships, sequences, and customization following SOLID principles.
"""

# ruff: noqa: D106
from __future__ import annotations

import uuid
from datetime import UTC, datetime

object

import factory
from factory import (
    Faker,
    LazyAttribute,
    Sequence,
)
from pydantic import BaseModel

from flext_core import FlextResult
from flext_core.constants import FlextConstants
from flext_core.models import FlextModels


# Base models for testing (these would typically come from domain models)
class TestUser(BaseModel):
    """Test user model for factory testing."""

    id: str
    name: str
    email: str
    age: int
    is_active: bool
    created_at: datetime
    metadata: dict[str, object]  # type: ignore[explicit-any]


class TestConfig(BaseModel):
    """Test configuration model for factory testing."""

    database_url: str
    log_level: str
    debug: bool
    timeout: int
    max_connections: int
    features: list[str]


class TestField(BaseModel):  # type: ignore[explicit-any]
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
    default_value: object = None  # type: ignore[explicit-any]
    pattern: str | None = None


# Factory Boy Factories
class UserFactory(factory.Factory):  # type: ignore[name-defined,misc]
    """Factory for creating test users with factory_boy."""

    class Meta:  # type: ignore[name-defined,misc]
        model = TestUser

    id = LazyAttribute(lambda _: str(uuid.uuid4()))  # type: ignore[no-untyped-call]
    name = Faker("name")  # type: ignore[no-untyped-call]
    email = Faker("email")  # type: ignore[no-untyped-call]
    age = Faker("random_int", min=18, max=80)  # type: ignore[no-untyped-call]
    is_active = True
    created_at = LazyAttribute(lambda _: datetime.now(UTC))  # type: ignore[no-untyped-call]
    metadata = factory.LazyFunction(  # type: ignore[attr-defined,no-untyped-call]
        lambda: {"department": "engineering", "level": "senior", "team": "backend"},
    )


class AdminUserFactory(UserFactory):
    """Factory for REDACTED_LDAP_BIND_PASSWORD users."""

    name = "Admin User"  # type: ignore[assignment]
    email = "REDACTED_LDAP_BIND_PASSWORD@example.com"  # type: ignore[assignment]
    metadata = factory.LazyFunction(  # type: ignore[attr-defined,no-untyped-call]
        lambda: {"department": "REDACTED_LDAP_BIND_PASSWORD", "level": "REDACTED_LDAP_BIND_PASSWORD", "permissions": ["all"]},
    )


class InactiveUserFactory(UserFactory):
    """Factory for inactive users."""

    is_active = False
    metadata = factory.LazyFunction(  # type: ignore[attr-defined,no-untyped-call]
        lambda: {
            "department": "archived",
            "level": "inactive",
            "archived_at": str(datetime.now(UTC)),
        },
    )


class ConfigFactory(factory.Factory):  # type: ignore[name-defined,misc]
    """Factory for creating test configurations."""

    class Meta:  # type: ignore[name-defined,misc]
        model = TestConfig

    database_url = "postgresql://test:test@localhost/test_db"
    log_level = "DEBUG"
    debug = True
    timeout = 30
    max_connections = 100
    features = factory.LazyFunction(lambda: ["auth", "cache", "metrics", "monitoring"])  # type: ignore[attr-defined,no-untyped-call]


class ProductionConfigFactory(ConfigFactory):
    """Factory for production-like configurations."""

    database_url = "postgresql://prod:prod@prod-db/prod_db"
    log_level = "INFO"
    debug = False
    timeout = 60
    max_connections = 500
    features = factory.LazyFunction(
        lambda: ["auth", "cache", "metrics", "monitoring", "alerts"]
    )  # type: ignore[attr-defined,no-untyped-call]


class StringFieldFactory(factory.Factory):  # type: ignore[name-defined,misc]
    """Factory for string field testing."""

    class Meta:  # type: ignore[name-defined,misc]
        model = TestField

    field_id = LazyAttribute(lambda _: str(uuid.uuid4()))  # type: ignore[no-untyped-call]
    field_name = Sequence(lambda n: f"string_field_{n}")  # type: ignore[no-untyped-call]
    field_type = FlextConstants.Enums.FieldType.STRING.value
    required = True
    description = LazyAttribute(lambda obj: f"Test string field: {obj.field_name}")  # type: ignore[no-untyped-call]
    min_length = 1
    max_length = 100
    pattern = r"^[a-zA-Z0-9_]+$"


class IntegerFieldFactory(factory.Factory):  # type: ignore[name-defined,misc]
    """Factory for integer field testing."""

    class Meta:  # type: ignore[name-defined,misc]
        model = TestField

    field_id = LazyAttribute(lambda _: str(uuid.uuid4()))  # type: ignore[no-untyped-call]
    field_name = Sequence(lambda n: f"integer_field_{n}")  # type: ignore[no-untyped-call]
    field_type = FlextConstants.Enums.FieldType.INTEGER.value
    required = True
    description = LazyAttribute(lambda obj: f"Test integer field: {obj.field_name}")  # type: ignore[no-untyped-call]
    min_value = 0
    max_value = 1000


class BooleanFieldFactory(factory.Factory):  # type: ignore[name-defined,misc]
    """Factory for boolean field testing."""

    class Meta:  # type: ignore[name-defined,misc]
        model = TestField

    field_id = LazyAttribute(lambda _: str(uuid.uuid4()))  # type: ignore[no-untyped-call]
    field_name = Sequence(lambda n: f"boolean_field_{n}")  # type: ignore[no-untyped-call]
    field_type = FlextConstants.Enums.FieldType.BOOLEAN.value
    required = True
    description = LazyAttribute(lambda obj: f"Test boolean field: {obj.field_name}")  # type: ignore[no-untyped-call]
    default_value = False


class FloatFieldFactory(factory.Factory):  # type: ignore[name-defined,misc]
    """Factory for float field testing."""

    class Meta:  # type: ignore[name-defined,misc]
        model = TestField

    field_id = LazyAttribute(lambda _: str(uuid.uuid4()))  # type: ignore[no-untyped-call]
    field_name = Sequence(lambda n: f"float_field_{n}")  # type: ignore[no-untyped-call]
    field_type = FlextConstants.Enums.FieldType.FLOAT.value
    required = True
    description = LazyAttribute(lambda obj: f"Test float field: {obj.field_name}")  # type: ignore[no-untyped-call]
    min_value = 0.0
    max_value = 1000.0


# FlextResult factories
class FlextResultFactory:
    """Factory for creating FlextResult instances for testing."""

    @staticmethod
    def success(data: object = None) -> FlextResult[object]:
        """Create successful FlextResult."""
        return FlextResult[object].ok(data or "test_success_data")

    @staticmethod
    def failure(
        error: str = "test_error", error_code: str = "TEST_ERROR"
    ) -> FlextResult[object]:  # type: ignore[explicit-any]
        """Create failed FlextResult."""
        return FlextResult[object].fail(error, error_code=error_code)

    @staticmethod
    def success_with_user() -> FlextResult[TestUser]:
        """Create successful FlextResult with user data."""
        user = UserFactory()
        return FlextResult[TestUser].ok(user)

    @staticmethod
    def success_with_config() -> FlextResult[TestConfig]:
        """Create successful FlextResult with config data."""
        config = ConfigFactory()
        return FlextResult[TestConfig].ok(config)


# Sequence generators for common patterns
class SequenceGenerators:
    """Generators for common sequence patterns in tests."""

    @staticmethod
    def entity_id_sequence() -> str:
        """Generate sequence of entity IDs."""
        return str(FlextModels.EntityId(root=f"test_entity_{uuid.uuid4()}"))

    @staticmethod
    def timestamp_sequence() -> FlextModels.Timestamp:
        """Generate sequence of timestamps."""
        return FlextModels.Timestamp(root=datetime.now(UTC))

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
        return UserFactory.create_batch(count)  # type: ignore[no-any-return]

    @staticmethod
    def create_mixed_users(count: int = 10) -> list[TestUser]:
        """Create batch of mixed user types."""
        users: list[TestUser] = []
        for i in range(count):
            if i % 3 == 0:
                users.append(AdminUserFactory())
            elif i % 5 == 0:
                users.append(InactiveUserFactory())
            else:
                users.append(UserFactory())
        return users

    @staticmethod
    def create_field_matrix() -> list[TestField]:
        """Create comprehensive field test matrix."""
        fields = []
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
    def empty_values() -> list[object]:  # type: ignore[explicit-any]
        """Generate empty/null test values."""
        return ["", [], {}, None, 0, False]

    @staticmethod
    def large_values() -> list[object]:  # type: ignore[explicit-any]
        """Generate large value test cases."""
        return [
            "x" * 10000,
            list(range(1000)),
            {f"key_{i}": f"value_{i}" for i in range(100)},
        ]


# Utility functions for common test patterns
def create_test_hierarchy() -> dict[str, object]:  # type: ignore[explicit-any]
    """Create hierarchical test data structure."""
    return {
        "root": UserFactory(),
        "children": UserFactory.create_batch(3),
        "REDACTED_LDAP_BIND_PASSWORD": AdminUserFactory(),
        "config": ConfigFactory(),
        "fields": BatchFactories.create_field_matrix(),
    }


def create_validation_test_cases() -> list[dict[str, object]]:  # type: ignore[explicit-any]
    """Create comprehensive validation test cases."""
    return [
        {
            "name": "valid_user",
            "data": UserFactory(),
            "expected_valid": True,
        },
        {
            "name": "invalid_email",
            "data": UserFactory(email="invalid-email"),
            "expected_valid": False,
        },
        {
            "name": "negative_age",
            "data": UserFactory(age=-5),
            "expected_valid": False,
        },
        {
            "name": "unicode_name",
            "data": UserFactory(name="测试用户"),
            "expected_valid": True,
        },
    ]


# Export commonly used factories and utilities
__all__ = [
    "AdminUserFactory",
    "BatchFactories",
    "BooleanFieldFactory",
    "ConfigFactory",
    "EdgeCaseGenerators",
    # Result factories
    "FlextResultFactory",
    "FloatFieldFactory",
    "InactiveUserFactory",
    "IntegerFieldFactory",
    "ProductionConfigFactory",
    # Utility classes
    "SequenceGenerators",
    "StringFieldFactory",
    # Main factories
    "UserFactory",
    # Utility functions
    "create_test_hierarchy",
    "create_validation_test_cases",
]
