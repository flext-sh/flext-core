"""Advanced test factories using factory_boy for consistent test data generation.

Provides factory_boy-based factories for creating test objects with proper
relationships, sequences, and customization following SOLID principles.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""
# mypy: disable-error-code="var-annotated,valid-type,no-untyped-call,name-defined,misc,no-any-return"

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import cast

import factory
from factory.declarations import LazyAttribute, LazyFunction, Sequence
from factory.faker import Faker

from flext_core import (
    FlextConstants,
    FlextResult,
    FlextTypes,
)
from flext_tests.domains import FlextTestsDomains


class FlextTestsFactories:
    """Unified test factories for FLEXT ecosystem.

    Consolidates all factory patterns into a single class interface.
    Uses factory_boy for consistent test data generation with proper
    relationships, sequences, and customization.
    """

    AuditLogValue = (
        str
        | int
        | bool
        | None
        | dict[str, str | int | bool | None]
        | FlextTypes.Core.StringList
    )
    AuditLog = dict[str, AuditLogValue]

    # === Factory Boy Factories ===

    class UserFactory(factory.Factory[FlextTestsDomains.TestUser]):
        """Factory for creating test users with factory_boy."""

        class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]:
            """Factory meta compatibility."""

            model = FlextTestsDomains.TestUser

        id = LazyAttribute(lambda _: str(uuid.uuid4()))
        name = Faker("name")
        email = Faker("email")
        age = Faker("random_int", min=18, max=80)
        is_active: bool = True
        created_at = LazyAttribute(lambda _: datetime.now(UTC))
        metadata = LazyFunction(
            lambda: {"department": "engineering", "level": "senior", "team": "backend"}
        )

        @classmethod
        def batch(cls, count: int = 10, **kwargs: object) -> list[dict[str, object]]:
            """Ultra-simple alias for test compatibility - create batch of users as dicts."""
            users = cls.create_batch(count, **kwargs)
            return [cls._ensure_dict(user) for user in users]

        @classmethod
        def create(cls, **kwargs: object) -> dict[str, object]:
            """Ultra-simple alias for test compatibility - create single user as dict."""
            user = super().create(**kwargs)
            return cls._ensure_dict(user)

        @classmethod
        def create_object(cls, **kwargs: object) -> object:
            """Create single user as object (TestUser) for tests that need attribute access."""
            return super().create(**kwargs)

        @staticmethod
        def _ensure_dict(user: object) -> dict[str, object]:
            """Ensure user data is in dictionary format, handling both dict and TestUser objects."""
            if isinstance(user, dict):
                # Already a dict - ensure correct keys
                return {
                    "id": user.get("id", ""),
                    "name": user.get("name", ""),
                    "email": user.get("email", ""),
                    "age": user.get("age", 0),
                    "active": user.get("active", user.get("is_active", True)),
                    "created_at": user.get("created_at", ""),
                }
            # TestUser object - convert to dict
            return {
                "id": getattr(user, "id", ""),
                "name": getattr(user, "name", ""),
                "email": getattr(user, "email", ""),
                "age": getattr(user, "age", 0),
                "active": getattr(user, "is_active", True),
                "created_at": getattr(user, "created_at", "").isoformat()
                if hasattr(getattr(user, "created_at", ""), "isoformat")
                else str(getattr(user, "created_at", "")),
            }

    class AdminUserFactory(factory.Factory[FlextTestsDomains.TestUser]):
        """Factory for REDACTED_LDAP_BIND_PASSWORD users."""

        class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]:
            """Factory meta compatibility."""

            model = FlextTestsDomains.TestUser

        id = LazyAttribute(lambda _: str(uuid.uuid4()))
        name = "Admin User"
        email = "REDACTED_LDAP_BIND_PASSWORD@example.com"
        age = Faker("random_int", min=25, max=60)
        is_active: bool = True
        created_at = LazyAttribute(lambda _: datetime.now(UTC))
        metadata = LazyFunction(
            lambda: {"department": "REDACTED_LDAP_BIND_PASSWORD", "level": "REDACTED_LDAP_BIND_PASSWORD", "permissions": "all"},
        )

        @classmethod
        def create_object(cls, **kwargs: object) -> object:
            """Create single REDACTED_LDAP_BIND_PASSWORD as object (TestUser) for tests that need attribute access."""
            return super().create(**kwargs)

    class InactiveUserFactory(factory.Factory[FlextTestsDomains.TestUser]):
        """Factory for inactive users."""

        class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]:
            """Factory meta compatibility."""

            model = FlextTestsDomains.TestUser

        id = LazyAttribute(lambda _: str(uuid.uuid4()))
        name = Faker("name")
        email = Faker("email")
        age = Faker("random_int", min=18, max=80)
        is_active = False
        created_at = LazyAttribute(lambda _: datetime.now(UTC))
        metadata = LazyFunction(
            lambda: {
                "department": "archived",
                "level": "inactive",
                "archived_at": str(datetime.now(UTC)),
            },
        )

    class ConfigFactory(factory.Factory[FlextTestsDomains.TestConfig]):
        """Factory for creating test configurations."""

        class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]:
            """Factory meta compatibility."""

            model = FlextTestsDomains.TestConfig

        database_url: str = "postgresql://test:test@localhost/test_db"
        log_level: str = "DEBUG"
        debug: bool = True
        timeout: int = 30
        max_connections: int = 100
        features = LazyFunction(lambda: ["auth", "cache", "metrics", "monitoring"])

    class ProductionConfigFactory(factory.Factory[FlextTestsDomains.TestConfig]):
        """Factory for production-like configurations."""

        class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]:
            """Factory meta compatibility."""

            model = FlextTestsDomains.TestConfig

        database_url = "postgresql://prod:prod@prod-db/prod_db"
        log_level = "INFO"
        debug = False
        timeout = 60
        max_connections = 500
        features = LazyFunction(
            lambda: ["auth", "cache", "metrics", "monitoring", "alerts"],
        )

    class StringFieldFactory(factory.Factory[FlextTestsDomains.TestField]):
        """Factory for string field testing."""

        class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]:
            """Factory meta compatibility."""

            model = FlextTestsDomains.TestField

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

    class IntegerFieldFactory(factory.Factory[FlextTestsDomains.TestField]):
        """Factory for integer field testing."""

        class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]:
            """Factory meta compatibility."""

            model = FlextTestsDomains.TestField

        field_id = LazyAttribute(lambda _: str(uuid.uuid4()))
        field_name = Sequence(lambda n: f"integer_field_{n}")
        field_type: str = FlextConstants.Enums.FieldType.INTEGER.value
        required: bool = True
        description = LazyAttribute(
            lambda obj: f"Test integer field: {getattr(obj, 'field_name', 'unknown')}",
        )
        min_value = 0
        max_value = 1000

    class BooleanFieldFactory(factory.Factory[FlextTestsDomains.TestField]):
        """Factory for boolean field testing."""

        class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]:
            """Factory meta compatibility."""

            model = FlextTestsDomains.TestField

        field_id = LazyAttribute(lambda _: str(uuid.uuid4()))
        field_name = Sequence(lambda n: f"boolean_field_{n}")
        field_type = FlextConstants.Enums.FieldType.BOOLEAN.value
        required = True
        description = LazyAttribute(
            lambda obj: f"Test boolean field: {getattr(obj, 'field_name', 'unknown')}",
        )
        default_value = False

    class FloatFieldFactory(factory.Factory[FlextTestsDomains.TestField]):
        """Factory for float field testing."""

        class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]:
            """Factory meta compatibility."""

            model = FlextTestsDomains.TestField

        field_id = LazyAttribute(lambda _: str(uuid.uuid4()))
        field_name = Sequence(lambda n: f"float_field_{n}")
        field_type = FlextConstants.Enums.FieldType.FLOAT.value
        required = True
        description = LazyAttribute(
            lambda obj: f"Test float field: {getattr(obj, 'field_name', 'unknown')}",
        )
        min_value = 0.0
        max_value = 1000.0

    class TestEntityFactory(factory.Factory[FlextTestsDomains.BaseTestEntity]):
        """Factory for creating test entities."""

        class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]:
            """Factory meta compatibility."""

            model = FlextTestsDomains.BaseTestEntity

        id = LazyAttribute(lambda _: str(uuid.uuid4()))
        name = Faker("company")
        created_at = LazyAttribute(lambda _: datetime.now(UTC))
        updated_at = LazyAttribute(lambda _: datetime.now(UTC))
        version = 1

        @classmethod
        def create(cls, **kwargs: object) -> FlextTestsDomains.BaseTestEntity:
            """Create a single test entity."""
            return super().create(**kwargs)

        @classmethod
        def create_batch(
            cls, size: int, **kwargs: object
        ) -> list[FlextTestsDomains.BaseTestEntity]:
            """Create a batch of test entities."""
            return super().create_batch(size, **kwargs)

    class TestValueObjectFactory(
        factory.Factory[FlextTestsDomains.BaseTestValueObject]
    ):
        """Factory for creating test value objects."""

        class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]:
            """Factory meta compatibility."""

            model = FlextTestsDomains.BaseTestValueObject

        value = Faker("word")
        description = Faker("sentence")
        category = Faker("word")

        @classmethod
        def create(cls, **kwargs: object) -> FlextTestsDomains.BaseTestValueObject:
            """Create a single test value object."""
            return super().create(**kwargs)

    # === FlextResult Factories ===

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
        def success_with_user() -> FlextResult[FlextTestsDomains.TestUser]:
            """Create successful FlextResult with user data."""
            user = cast("FlextTestsDomains.TestUser", FlextTestsFactories.UserFactory())
            return FlextResult[FlextTestsDomains.TestUser].ok(user)

        @staticmethod
        def success_with_config() -> FlextResult[FlextTestsDomains.TestConfig]:
            """Create successful FlextResult with config data."""
            config = cast(
                "FlextTestsDomains.TestConfig", FlextTestsFactories.ConfigFactory()
            )
            return FlextResult[FlextTestsDomains.TestConfig].ok(config)

        @staticmethod
        def create_success(data: object = None) -> FlextResult[object]:
            """Create successful FlextResult (alias for success method)."""
            return FlextTestsFactories.FlextResultFactory.success(data)

        @staticmethod
        def create_failure(
            error: str = "test_error",
            error_code: str = "TEST_ERROR",
        ) -> FlextResult[object]:
            """Create failed FlextResult (alias for failure method)."""
            return FlextTestsFactories.FlextResultFactory.failure(error, error_code)

        @staticmethod
        def create_batch(size: int = 10) -> list[FlextResult[object]]:
            """Create batch of successful FlextResults."""
            return [
                FlextTestsFactories.FlextResultFactory.success(f"test_data_{i}")
                for i in range(size)
            ]

    # === Sequence Generators ===

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

    # === Batch Factories ===

    class BatchFactories:
        """Batch creation utilities for performance and integration testing."""

        @staticmethod
        def create_users(count: int = 10) -> list[FlextTestsDomains.TestUser]:
            """Create batch of test users."""
            return FlextTestsFactories.UserFactory.create_batch(count)

        @staticmethod
        def create_mixed_users(count: int = 10) -> list[FlextTestsDomains.TestUser]:
            """Create batch of mixed user types."""
            users: list[FlextTestsDomains.TestUser] = []
            for i in range(count):
                if i % 3 == 0:
                    users.append(
                        cast(
                            "FlextTestsDomains.TestUser",
                            FlextTestsFactories.AdminUserFactory(),
                        )
                    )
                elif i % 5 == 0:
                    users.append(
                        cast(
                            "FlextTestsDomains.TestUser",
                            FlextTestsFactories.InactiveUserFactory(),
                        )
                    )
                else:
                    users.append(
                        cast(
                            "FlextTestsDomains.TestUser",
                            FlextTestsFactories.UserFactory(),
                        )
                    )
            return users

        @staticmethod
        def create_field_matrix() -> list[FlextTestsDomains.TestField]:
            """Create comprehensive field test matrix."""
            fields: list[FlextTestsDomains.TestField] = []
            fields.extend(FlextTestsFactories.StringFieldFactory.create_batch(3))
            fields.extend(FlextTestsFactories.IntegerFieldFactory.create_batch(3))
            fields.extend(FlextTestsFactories.BooleanFieldFactory.create_batch(2))
            fields.extend(FlextTestsFactories.FloatFieldFactory.create_batch(2))
            return fields

    # === Edge Case Generators ===

    class EdgeCaseGenerators:
        """Generators for edge case testing scenarios."""

        @staticmethod
        def unicode_strings() -> FlextTypes.Core.StringList:
            """Generate unicode test strings."""
            return ["🚀", "测试", "مرحبا", "🔥🎯", "Ñoño", "café"]

        @staticmethod
        def special_characters() -> FlextTypes.Core.StringList:
            """Generate special character test strings."""
            return ["!@#$%^&*()", "\n\t\r", "\\", "'\"", "<script>", "SELECT * FROM"]

        @staticmethod
        def boundary_numbers() -> list[int | float]:
            """Generate boundary number test values."""
            return [0, -1, 1, 999999999, -999999999, 1e-10, float("inf"), float("-inf")]

        @staticmethod
        def empty_values() -> FlextTypes.Core.List:
            """Generate empty/null test values."""
            return ["", [], {}, None, 0, False]

        @staticmethod
        def large_values() -> FlextTypes.Core.List:
            """Generate large value test cases."""
            return [
                "x" * 10000,
                list(range(1000)),
                {f"key_{i}": f"value_{i}" for i in range(100)},
            ]

    # === Utility Functions ===

    @staticmethod
    def create_test_hierarchy() -> FlextTypes.Core.Dict:
        """Create hierarchical test data structure."""
        return {
            "root": cast(
                "FlextTestsDomains.TestUser", FlextTestsFactories.UserFactory()
            ),
            "children": FlextTestsFactories.UserFactory.create_batch(3),
            "REDACTED_LDAP_BIND_PASSWORD": cast(
                "FlextTestsDomains.TestUser", FlextTestsFactories.AdminUserFactory()
            ),
            "config": cast(
                "FlextTestsDomains.TestConfig", FlextTestsFactories.ConfigFactory()
            ),
            "fields": FlextTestsFactories.BatchFactories.create_field_matrix(),
        }

    @staticmethod
    def create_validation_test_cases() -> list[FlextTypes.Core.Dict]:
        """Create comprehensive validation test cases."""
        return [
            {
                "name": "valid_user",
                "data": cast(
                    "FlextTestsDomains.TestUser", FlextTestsFactories.UserFactory()
                ),
                "expected_valid": True,
            },
            {
                "name": "invalid_email",
                "data": FlextTestsDomains.TestUser(
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
                "data": FlextTestsDomains.TestUser(
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
                "data": FlextTestsDomains.TestUser(
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

    # === Convenience Functions ===

    @staticmethod
    def success_result(data: object = "test_data") -> FlextResult[object]:
        """Create successful FlextResult."""
        return FlextResult[object].ok(data)

    @staticmethod
    def failure_result(
        error: str = "Test error",
        error_code: str = "TEST_ERROR",
    ) -> FlextResult[object]:
        """Create failed FlextResult."""
        return FlextResult[object].fail(error, error_code=error_code)

    @staticmethod
    def validation_failure(field: str = "test_field") -> FlextResult[object]:
        """Create validation failure FlextResult."""
        return FlextResult[object].fail(
            f"Validation failed for field: {field}",
            error_code="VALIDATION_ERROR",
        )


# === REMOVED COMPATIBILITY ALIASES AND FACADES ===
# Legacy compatibility removed as per user request
# All compatibility facades, aliases and protocol facades have been commented out
# Only FlextTestsFactories class is now exported

# Main class alias for backward compatibility - REMOVED
# FlextTestsFactory = FlextTestsFactories

# Legacy UserFactory class - REMOVED (commented out)
# class UserFactory:
#     """Compatibility facade for UserFactory - use FlextTestsFactories instead."""
#     ... all methods commented out

# Legacy AdminUserFactory class - REMOVED (commented out)
# class AdminUserFactory:
#     """Compatibility facade for AdminUserFactory - use FlextTestsFactories instead."""
#     ... all methods commented out

# Legacy InactiveUserFactory class - REMOVED (commented out)
# class InactiveUserFactory:
#     """Compatibility facade for InactiveUserFactory - use FlextTestsFactories instead."""
#     ... all methods commented out

# Legacy ConfigFactory class - REMOVED (commented out)
# class ConfigFactory:
#     """Compatibility facade for ConfigFactory - use FlextTestsFactories instead."""
#     ... all methods commented out

# Legacy ProductionConfigFactory class - REMOVED (commented out)
# class ProductionConfigFactory:
#     """Compatibility facade for ProductionConfigFactory - use FlextTestsFactories instead."""
#     ... all methods commented out

# Legacy field factory classes - REMOVED (commented out)
# class StringFieldFactory, IntegerFieldFactory, BooleanFieldFactory, FloatFieldFactory:
#     ... all methods commented out

# Legacy TestEntityFactory class - REMOVED (commented out)
# class TestEntityFactory:
#     """Compatibility facade for TestEntityFactory - use FlextTestsFactories instead."""
#     ... all methods commented out

# Legacy TestValueObjectFactory class - REMOVED (commented out)
# class TestValueObjectFactory:
#     """Compatibility facade for TestValueObjectFactory - use FlextTestsFactories instead."""
#     ... all methods commented out

# Legacy FlextResultFactory class - REMOVED (commented out)
# class FlextResultFactory:
#     """Compatibility facade for FlextResultFactory - use FlextTestsFactories instead."""
#     ... all methods commented out

# Legacy SequenceGenerators class - REMOVED (commented out)
# class SequenceGenerators:
#     """Compatibility facade for SequenceGenerators - use FlextTestsFactories instead."""
#     ... all methods commented out

# Legacy BatchFactories class - REMOVED (commented out)
# class BatchFactories:
#     """Compatibility facade for BatchFactories - use FlextTestsFactories instead."""
#     ... all methods commented out

# Legacy EdgeCaseGenerators class - REMOVED (commented out)
# class EdgeCaseGenerators:
#     """Compatibility facade for EdgeCaseGenerators - use FlextTestsFactories instead."""
#     ... all methods commented out

# Convenience functions for backward compatibility - REMOVED (commented out)
# def success_result, failure_result, validation_failure, etc.
#     ... all functions commented out

# Export only the unified class
__all__ = [
    "FlextTestsFactories",
]
