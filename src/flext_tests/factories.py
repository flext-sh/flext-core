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
        def create(**overrides: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Create user data with optional overrides.

            Returns:
                FlextTypes.Core.Dict: User data dictionary with generated values

            """
            return FlextTestsDomains.create_user(**overrides)

        # build method removed - use create directly per FLEXT architectural principles

        @staticmethod
        def create_batch(
            size: int, **kwargs: FlextTypes.Core.Dict
        ) -> list[FlextTypes.Core.Dict]:
            """Create batch of users.

            Returns:
                list[FlextTypes.Core.Dict]: List of user data dictionaries

            """
            return [FlextTestsDomains.create_user(**kwargs) for _ in range(size)]

        @staticmethod
        def REDACTED_LDAP_BIND_PASSWORD_user(**overrides: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Create REDACTED_LDAP_BIND_PASSWORD user.

            Returns:
                FlextTypes.Core.Dict: Admin user data dictionary

            """
            defaults: FlextTypes.Core.Dict = {
                "name": "Admin User",
                "email": "REDACTED_LDAP_BIND_PASSWORD@company.com",
                "age": 35,
            }
            defaults.update(overrides)
            return FlextTestsDomains.create_user(**defaults)

        @staticmethod
        def inactive_user(**overrides: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Create inactive user.

            Returns:
                FlextTypes.Core.Dict: Inactive user data dictionary

            """
            defaults: FlextTypes.Core.Dict = {"active": False, "name": "Inactive User"}
            defaults.update(overrides)
            return FlextTestsDomains.create_user(**defaults)

        # Alias methods removed - use REDACTED_LDAP_BIND_PASSWORD_user, inactive_user, create_batch directly
        # per FLEXT architectural principles

    # =========================================================================
    # CONFIG FACTORIES - Configuration data creation
    # =========================================================================

    class ConfigFactory:
        """Simple configuration factory."""

        @staticmethod
        def create(**overrides: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Create configuration data.

            Returns:
                FlextTypes.Core.Dict: Configuration data dictionary

            """
            return FlextTestsDomains.create_configuration(**overrides)

        @staticmethod
        def production_config(
            **overrides: FlextTypes.Core.Dict,
        ) -> FlextTypes.Core.Dict:
            """Create production configuration.

            Returns:
                FlextTypes.Core.Dict: Production configuration data dictionary

            """
            defaults: FlextTypes.Core.Dict = {
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
        def string_field(**overrides: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Create string field definition.

            Returns:
                FlextTypes.Core.Dict: String field definition dictionary

            """
            defaults: FlextTypes.Core.Dict = {
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
        def integer_field(**overrides: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Create integer field definition.

            Returns:
                FlextTypes.Core.Dict: Integer field definition dictionary

            """
            defaults: FlextTypes.Core.Dict = {
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
        def boolean_field(**overrides: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Create boolean field definition.

            Returns:
                FlextTypes.Core.Dict: Boolean field definition dictionary

            """
            defaults: FlextTypes.Core.Dict = {
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
        def float_field(**overrides: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Create float field definition.

            Returns:
                FlextTypes.Core.Dict: Float field definition dictionary

            """
            defaults: FlextTypes.Core.Dict = {
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
            field_type: str,
            size: int,
            **kwargs: FlextTypes.Core.Dict,
        ) -> list[FlextTypes.Core.Dict]:
            """Create batch of fields by type.

            Returns:
                list[FlextTypes.Core.Dict]: List of field definition dictionaries

            """
            method_map = {
                "string": FlextTestsFactories.FieldFactory.string_field,
                "integer": FlextTestsFactories.FieldFactory.integer_field,
                "boolean": FlextTestsFactories.FieldFactory.boolean_field,
                "float": FlextTestsFactories.FieldFactory.float_field,
            }

            field_method = method_map.get(
                field_type,
                FlextTestsFactories.FieldFactory.string_field,
            )
            return [field_method(**kwargs) for _ in range(size)]

    # =========================================================================
    # ENTITY FACTORIES - Domain entity creation
    # =========================================================================

    class EntityFactory:
        """Simple entity factory for domain testing."""

        @staticmethod
        def test_entity(**overrides: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Create test entity.

            Returns:
                FlextTypes.Core.Dict: Test entity data dictionary

            """
            defaults: FlextTypes.Core.Dict = {
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
        def test_value_object(
            **overrides: FlextTypes.Core.Dict,
        ) -> FlextTypes.Core.Dict:
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
            self,
            *,
            production: bool = False,
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
            self,
            count: int = 5,
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

        # Alias methods removed - use success_result, failure_result directly
        # per FLEXT architectural principles

    # =========================================================================
    # SERVICE FACTORIES - Service layer testing
    # =========================================================================

    class ServiceFactory:
        """Simple service factory for testing service layer."""

        @staticmethod
        def create(**overrides: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Create service data using the ``FlextTypes.Core.Dict`` alias."""
            defaults: FlextTypes.Core.Dict = {
                "service_id": str(uuid.uuid4()),
                "service_name": "Test Service",
                "version": "1.0.0",
                "status": "active",
                "endpoints": ["/api/test"],
                "metadata": {},
            }
            defaults.update(overrides)
            return defaults

        @staticmethod
        def create_batch(
            size: int, **kwargs: FlextTypes.Core.Dict
        ) -> list[FlextTypes.Core.Dict]:
            """Create service data batch using ``list[FlextTypes.Core.Dict]``."""
            return [
                FlextTestsFactories.ServiceFactory.create(**kwargs) for _ in range(size)
            ]

    # =========================================================================
    # MOCK FACTORIES - Mock data for testing
    # =========================================================================

    class MockFactory:
        """Simple mock factory for testing."""

        @staticmethod
        def create(**overrides: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Create mock payload using the ``FlextTypes.Core.Dict`` alias."""
            defaults: FlextTypes.Core.Dict = {
                "mock_id": str(uuid.uuid4()),
                "mock_type": "test_mock",
                "data": {"key": "value"},
                "created_at": datetime.now(UTC).isoformat(),
            }
            defaults.update(overrides)
            return defaults

        @staticmethod
        def create_batch(
            size: int, **kwargs: FlextTypes.Core.Dict
        ) -> list[FlextTypes.Core.Dict]:
            """Create mock payload batch using ``list[FlextTypes.Core.Dict]``."""
            return [
                FlextTestsFactories.MockFactory.create(**kwargs) for _ in range(size)
            ]

    # =========================================================================
    # SEQUENCE FACTORIES - Sequential data generation
    # =========================================================================

    class SequenceFactory:
        """Simple sequence factory for sequential data."""

        @staticmethod
        def create(**overrides: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Create sequence data."""
            defaults: FlextTypes.Core.Dict = {
                "sequence_id": str(uuid.uuid4()),
                "sequence_name": "Test Sequence",
                "current_value": 1,
                "step": 1,
                "max_value": 1000,
            }
            defaults.update(overrides)
            return defaults

        @staticmethod
        def create_batch(
            size: int, **kwargs: FlextTypes.Core.Dict
        ) -> list[FlextTypes.Core.Dict]:
            """Create batch of sequences."""
            return [
                FlextTestsFactories.SequenceFactory.create(**kwargs)
                for _ in range(size)
            ]

    # =========================================================================
    # POST FACTORIES - Post/Article data for testing
    # =========================================================================

    class PostFactory:
        """Simple post factory for testing post/article functionality."""

        @staticmethod
        def create(**overrides: object) -> dict[str, object]:
            """Create post data."""
            defaults: dict[str, object] = {
                "post_id": str(uuid.uuid4()),
                "title": "Test Post",
                "content": "This is test content for the post.",
                "author": "Test Author",
                "published": True,
                "created_at": datetime.now(UTC).isoformat(),
                "tags": ["test", "post"],
            }
            defaults.update(overrides)
            return defaults

        @staticmethod
        def create_batch(
            size: int, **kwargs: FlextTypes.Core.Dict
        ) -> list[FlextTypes.Core.Dict]:
            """Create batch of posts."""
            return [
                FlextTestsFactories.PostFactory.create(**kwargs) for _ in range(size)
            ]

        @staticmethod
        def draft_post(**overrides: dict[str, object]) -> dict[str, object]:
            """Create draft post."""
            defaults: dict[str, object] = {"published": False, "title": "Draft Post"}
            defaults.update(overrides)
            return FlextTestsFactories.PostFactory.create(**defaults)

        @staticmethod
        def published_post(**overrides: dict[str, object]) -> dict[str, object]:
            """Create published post."""
            defaults: dict[str, object] = {
                "published": True,
                "title": "Published Post",
            }
            defaults.update(overrides)
            return FlextTestsFactories.PostFactory.create(**defaults)

    # =========================================================================
    # ALIASES FOR BACKWARD COMPATIBILITY
    # =========================================================================

    # Aliases removed - use direct class access per FLEXT architectural principles
    # Use UserFactory, ConfigFactory, ResultFactory, ServiceFactory, MockFactory,
    # SequenceFactory, PostFactory directly


# Export main factory class
__all__ = ["FlextTestsFactories"]
