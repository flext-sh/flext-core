"""Simple test factories - Clean, type-safe, no over-engineering.

Uses FlextTestsDomains for consistent test data generation without factory_boy complications.
Follows the audit recommendation: remove what's wrong/over-engineered.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from flext_core import (
    FlextResult,
    FlextTypes,
)
from flext_tests.domains import FlextTestsDomains


class FlextTestsFactories:
    """Simple, clean test data factories without factory_boy over-engineering.

    Consolidates test data creation into simple, type-safe factory methods.
    Uses FlextTestsDomains for underlying data generation.
    """

    # =========================================================================
    # USER FACTORIES - Simple user data creation
    # =========================================================================

    class UserFactory:
        """Simple user factory using FlextTestsDomains."""

        @staticmethod
        def create(**overrides: object) -> FlextTypes.Core.Dict:
            """Create user data with optional overrides."""
            return FlextTestsDomains.create_user(**overrides)

        @staticmethod
        def build(**overrides: object) -> FlextTypes.Core.Dict:
            """Build user data (alias for create)."""
            return FlextTestsDomains.create_user(**overrides)

        @staticmethod
        def create_batch(size: int, **kwargs: object) -> list[FlextTypes.Core.Dict]:
            """Create batch of users."""
            return [FlextTestsDomains.create_user(**kwargs) for _ in range(size)]

        @staticmethod
        def REDACTED_LDAP_BIND_PASSWORD_user(**overrides: object) -> FlextTypes.Core.Dict:
            """Create REDACTED_LDAP_BIND_PASSWORD user."""
            defaults = {"name": "Admin User", "email": "REDACTED_LDAP_BIND_PASSWORD@company.com", "age": 35}
            defaults.update(overrides)
            return FlextTestsDomains.create_user(**defaults)

        @staticmethod
        def inactive_user(**overrides: object) -> FlextTypes.Core.Dict:
            """Create inactive user."""
            defaults = {"active": False, "name": "Inactive User"}
            defaults.update(overrides)
            return FlextTestsDomains.create_user(**defaults)

    # =========================================================================
    # CONFIG FACTORIES - Configuration data creation
    # =========================================================================

    class ConfigFactory:
        """Simple configuration factory."""

        @staticmethod
        def create(**overrides: object) -> FlextTypes.Core.Dict:
            """Create configuration data."""
            return FlextTestsDomains.create_configuration(**overrides)

        @staticmethod
        def production_config(**overrides: object) -> FlextTypes.Core.Dict:
            """Create production configuration."""
            defaults = {
                "debug": False,
                "log_level": "ERROR",
                "cache_enabled": True,
            }
            defaults.update(overrides)
            return FlextTestsDomains.create_configuration(**defaults)

    # =========================================================================
    # FIELD FACTORIES - Field data for validation testing
    # =========================================================================

    class FieldFactory:
        """Simple field factory for validation testing."""

        @staticmethod
        def string_field(**overrides: object) -> FlextTypes.Core.Dict:
            """Create string field definition."""
            defaults = {
                "field_id": str(uuid.uuid4()),
                "field_name": "test_string_field",
                "field_type": "string",
                "required": True,
                "description": "Test string field",
                "min_length": 1,
                "max_length": 100,
            }
            defaults.update(overrides)
            return defaults

        @staticmethod
        def integer_field(**overrides: object) -> FlextTypes.Core.Dict:
            """Create integer field definition."""
            defaults = {
                "field_id": str(uuid.uuid4()),
                "field_name": "test_integer_field",
                "field_type": "integer",
                "required": True,
                "description": "Test integer field",
                "min_value": 0,
                "max_value": 1000,
            }
            defaults.update(overrides)
            return defaults

        @staticmethod
        def boolean_field(**overrides: object) -> FlextTypes.Core.Dict:
            """Create boolean field definition."""
            defaults = {
                "field_id": str(uuid.uuid4()),
                "field_name": "test_boolean_field",
                "field_type": "boolean",
                "required": False,
                "description": "Test boolean field",
                "default_value": False,
            }
            defaults.update(overrides)
            return defaults

        @staticmethod
        def float_field(**overrides: object) -> FlextTypes.Core.Dict:
            """Create float field definition."""
            defaults = {
                "field_id": str(uuid.uuid4()),
                "field_name": "test_float_field",
                "field_type": "float",
                "required": True,
                "description": "Test float field",
                "min_value": 0.0,
                "max_value": 100.0,
            }
            defaults.update(overrides)
            return defaults

        @staticmethod
        def create_batch(
            field_type: str, size: int, **kwargs: object
        ) -> list[FlextTypes.Core.Dict]:
            """Create batch of fields by type."""
            method_map = {
                "string": FlextTestsFactories.FieldFactory.string_field,
                "integer": FlextTestsFactories.FieldFactory.integer_field,
                "boolean": FlextTestsFactories.FieldFactory.boolean_field,
                "float": FlextTestsFactories.FieldFactory.float_field,
            }

            field_method = method_map.get(
                field_type, FlextTestsFactories.FieldFactory.string_field
            )
            return [field_method(**kwargs) for _ in range(size)]

    # =========================================================================
    # ENTITY FACTORIES - Domain entity creation
    # =========================================================================

    class EntityFactory:
        """Simple entity factory for domain testing."""

        @staticmethod
        def test_entity(**overrides: object) -> FlextTypes.Core.Dict:
            """Create test entity."""
            defaults = {
                "id": str(uuid.uuid4()),
                "name": "Test Entity",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "version": 1,
                "metadata": {},
            }
            defaults.update(overrides)
            return defaults

        @staticmethod
        def test_value_object(**overrides: object) -> FlextTypes.Core.Dict:
            """Create test value object."""
            defaults: FlextTypes.Core.Dict = {
                "value": "test_value",
                "description": "Test value object",
                "category": "test",
                "tags": ["test", "value_object"],
            }
            defaults.update(overrides)
            return defaults

    # =========================================================================
    # CONVENIENCE METHODS - Bulk operations and utilities
    # =========================================================================

    @staticmethod
    def create_test_dataset() -> FlextTypes.Core.Dict:
        """Create test dataset with all entity types."""
        return {
            "users": FlextTestsFactories.UserFactory.create_batch(5),
            "configs": [FlextTestsFactories.ConfigFactory.create()],
            "fields": FlextTestsFactories.FieldFactory.create_batch("string", 3),
            "entities": [FlextTestsFactories.EntityFactory.test_entity()],
        }

    @staticmethod
    def create_validation_test_data() -> FlextTypes.Core.Dict:
        """Create data specifically for validation testing."""
        return {
            "valid_emails": FlextTestsDomains.valid_email_cases(),
            "invalid_emails": FlextTestsDomains.invalid_email_cases(),
            "valid_ages": FlextTestsDomains.valid_ages(),
            "invalid_ages": FlextTestsDomains.invalid_ages(),
        }

    @staticmethod
    def create_realistic_test_data() -> FlextTypes.Core.Dict:
        """Create realistic test data for integration testing."""
        return {
            "user_registration": FlextTestsDomains.user_registration_data(),
            "order": FlextTestsDomains.order_data(),
            "api_response": FlextTestsDomains.api_response_data(),
        }

    # =========================================================================
    # BUILDER PATTERN - For complex test scenarios
    # =========================================================================

    class TestDataBuilder:
        """Builder pattern for complex test data scenarios."""

        def __init__(self) -> None:
            """Initialize test data builder."""
            self._data: FlextTypes.Core.Dict = {}

        def with_users(self, count: int = 3) -> FlextTestsFactories.TestDataBuilder:
            """Add users to test data."""
            self._data["users"] = FlextTestsFactories.UserFactory.create_batch(count)
            return self

        def with_configs(
            self, *, production: bool = False
        ) -> FlextTestsFactories.TestDataBuilder:
            """Add configuration to test data."""
            if production:
                self._data["config"] = (
                    FlextTestsFactories.ConfigFactory.production_config()
                )
            else:
                self._data["config"] = FlextTestsFactories.ConfigFactory.create()
            return self

        def with_validation_fields(
            self, count: int = 5
        ) -> FlextTestsFactories.TestDataBuilder:
            """Add validation fields to test data."""
            self._data["validation_fields"] = (
                FlextTestsFactories.FieldFactory.create_batch("string", count)
            )
            return self

        def build(self) -> FlextTypes.Core.Dict:
            """Build the final test data."""
            return self._data.copy()

    # =========================================================================
    # RESULT FACTORIES - For FlextResult testing
    # =========================================================================

    class ResultFactory:
        """Factory for creating FlextResult test instances."""

        @staticmethod
        def success_result(data: object = None) -> FlextResult[object]:
            """Create successful result."""
            return FlextResult[object].ok(data or {"success": True})

        @staticmethod
        def failure_result(error: str = "Test error") -> FlextResult[object]:
            """Create failure result."""
            return FlextResult[object].fail(error)

        @staticmethod
        def user_result(*, success: bool = True) -> FlextResult[FlextTypes.Core.Dict]:
            """Create user-specific result."""
            if success:
                user_data = FlextTestsFactories.UserFactory.create()
                return FlextResult[FlextTypes.Core.Dict].ok(user_data)
            return FlextResult[FlextTypes.Core.Dict].fail("User creation failed")


# Export main factory class
__all__ = ["FlextTestsFactories"]
