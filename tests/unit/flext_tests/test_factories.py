"""Unit tests for flext_tests.factories module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import pytest

from flext_core import FlextResult
from flext_tests.factories import FlextTestsFactories

# Access nested classes
User = FlextTestsFactories.User
Config = FlextTestsFactories.Config
Service = FlextTestsFactories.Service


class TestUser:
    """Test suite for User model."""

    def test_user_creation_default(self) -> None:
        """Test User model creation with defaults."""
        user = User(
            id="test-123",
            name="Test User",
            email="test@example.com",
        )

        assert user.id == "test-123"
        assert user.name == "Test User"
        assert user.email == "test@example.com"
        assert user.active is True

    def test_user_creation_with_active(self) -> None:
        """Test User model creation with active flag."""
        user = User(
            id="test-123",
            name="Test User",
            email="test@example.com",
            active=False,
        )

        assert user.active is False


class TestConfig:
    """Test suite for Config model."""

    def test_config_creation_default(self) -> None:
        """Test Config model creation with defaults."""
        config = Config()

        assert config.service_type == "api"
        assert config.environment == "test"
        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.timeout == 30
        assert config.max_retries == 3

    def test_config_creation_custom(self) -> None:
        """Test Config model creation with custom values."""
        config = Config(
            service_type="database",
            environment="production",
            debug=False,
            timeout=60,
        )

        assert config.service_type == "database"
        assert config.environment == "production"
        assert config.debug is False
        assert config.timeout == 60


class TestService:
    """Test suite for Service model."""

    def test_service_creation_minimal(self) -> None:
        """Test Service model creation with minimal fields."""
        service = Service(id="test-123")

        assert service.id == "test-123"
        assert service.type == "api"
        assert service.name == ""
        assert service.status == "active"

    def test_service_creation_complete(self) -> None:
        """Test Service model creation with all fields."""
        service = Service(
            id="test-123",
            type="database",
            name="Database Service",
            status="inactive",
        )

        assert service.id == "test-123"
        assert service.type == "database"
        assert service.name == "Database Service"
        assert service.status == "inactive"


class TestFlextTestsFactories:
    """Test suite for FlextTestsFactories class."""

    def test_create_user_default(self) -> None:
        """Test create_user with default parameters."""
        user = FlextTestsFactories.create_user()

        assert isinstance(user, User)
        assert user.id is not None
        assert user.name == "Test User"
        assert "@example.com" in user.email
        assert user.active is True

    def test_create_user_custom(self) -> None:
        """Test create_user with custom parameters."""
        user = FlextTestsFactories.create_user(
            user_id="custom-123",
            name="Custom User",
            email="custom@test.com",
        )

        assert user.id == "custom-123"
        assert user.name == "Custom User"
        assert user.email == "custom@test.com"

    def test_create_user_with_overrides(self) -> None:
        """Test create_user with overrides."""
        user = FlextTestsFactories.create_user(
            name="Base User",
            active=False,
        )

        assert user.name == "Base User"
        assert user.active is False

    def test_create_config_default(self) -> None:
        """Test create_config with default parameters."""
        config = FlextTestsFactories.create_config()

        assert isinstance(config, Config)
        assert config.service_type == "api"
        assert config.environment == "test"
        assert config.debug is True

    def test_create_config_custom(self) -> None:
        """Test create_config with custom parameters."""
        config = FlextTestsFactories.create_config(
            service_type="database",
            environment="production",
            debug=False,
            timeout=60,
        )

        assert config.service_type == "database"
        assert config.environment == "production"
        assert config.debug is False
        assert config.timeout == 60

    def test_create_config_with_overrides(self) -> None:
        """Test create_config with overrides."""
        config = FlextTestsFactories.create_config(
            log_level="INFO",
            max_retries=5,
        )

        assert config.log_level == "INFO"
        assert config.max_retries == 5

    def test_create_service_default(self) -> None:
        """Test create_service with default parameters."""
        service = FlextTestsFactories.create_service()

        assert isinstance(service, Service)
        assert service.id is not None
        assert service.type == "api"
        assert "Test api Service" in service.name
        assert service.status == "active"

    def test_create_service_custom(self) -> None:
        """Test create_service with custom parameters."""
        service = FlextTestsFactories.create_service(
            service_type="database",
            service_id="custom-123",
            name="Custom Service",
        )

        assert service.id == "custom-123"
        assert service.type == "database"
        assert service.name == "Custom Service"

    def test_create_service_with_overrides(self) -> None:
        """Test create_service with overrides."""
        service = FlextTestsFactories.create_service(
            status="inactive",
        )

        assert service.status == "inactive"

    def test_batch_users_default(self) -> None:
        """Test batch_users with default count."""
        users = FlextTestsFactories.batch_users()

        assert len(users) == 5
        assert all(isinstance(user, User) for user in users)
        assert users[0].name == "User 0"
        assert users[1].name == "User 1"

    def test_batch_users_custom_count(self) -> None:
        """Test batch_users with custom count."""
        users = FlextTestsFactories.batch_users(count=3)

        assert len(users) == 3
        assert all(isinstance(user, User) for user in users)

    def test_create_test_operation_simple(self) -> None:
        """Test create_test_operation with 'simple' type."""
        operation = FlextTestsFactories.create_test_operation("simple")

        assert callable(operation)
        result = operation()
        assert result == "success"

    def test_create_test_operation_add(self) -> None:
        """Test create_test_operation with 'add' type."""
        operation = FlextTestsFactories.create_test_operation("add")

        assert callable(operation)
        result = operation(2, 3)
        assert result == 5

    def test_create_test_operation_format(self) -> None:
        """Test create_test_operation with 'format' type."""
        operation = FlextTestsFactories.create_test_operation("format")

        assert callable(operation)
        result = operation("name", value=20)
        assert result == "name: 20"

    def test_create_test_operation_error(self) -> None:
        """Test create_test_operation with 'error' type."""
        operation = FlextTestsFactories.create_test_operation(
            "error",
            error_message="Custom error",
        )

        assert callable(operation)
        with pytest.raises(ValueError, match="Custom error"):
            operation()

    def test_create_test_operation_type_error(self) -> None:
        """Test create_test_operation with 'type_error' type."""
        operation = FlextTestsFactories.create_test_operation(
            "type_error",
            error_message="Type mismatch",
        )

        assert callable(operation)
        with pytest.raises(TypeError, match="Type mismatch"):
            operation()

    def test_create_test_operation_unknown(self) -> None:
        """Test create_test_operation with unknown type."""
        operation = FlextTestsFactories.create_test_operation("unknown")

        assert callable(operation)
        result = operation()
        assert result == "unknown operation: unknown"

    def test_create_test_service_default(self) -> None:
        """Test create_test_service with default type."""
        service_class = FlextTestsFactories.create_test_service()
        service = service_class()

        assert service.name is None
        assert service.amount is None
        assert service.enabled is None

        result = service.execute()
        assert result.is_success
        assert result.value == {"service_type": "test"}

    def test_create_test_service_user(self) -> None:
        """Test create_test_service with 'user' type."""
        service_class = FlextTestsFactories.create_test_service("user")
        service = service_class()

        result = service.execute()
        assert result.is_success
        assert "user_id" in result.value
        assert result.value["user_id"] == "test_123"

    def test_create_test_service_user_with_default(self) -> None:
        """Test create_test_service with 'user' type and default flag."""
        service_class = FlextTestsFactories.create_test_service("user", default=True)
        service = service_class()

        result = service.execute()
        assert result.is_success
        assert result.value["user_id"] == "default_123"

    def test_create_test_service_complex_valid(self) -> None:
        """Test create_test_service with 'complex' type and valid data."""
        service_class = FlextTestsFactories.create_test_service("complex")
        service = service_class(name="Test", amount=100, enabled=True)

        result = service.execute()
        assert result.is_success
        assert result.value == {"result": "success"}

    def test_create_test_service_complex_empty_name(self) -> None:
        """Test create_test_service with 'complex' type and empty name."""
        service_class = FlextTestsFactories.create_test_service("complex")
        service = service_class(name="")

        result = service.execute()
        assert result.is_failure
        assert "Name is required" in result.error

    def test_create_test_service_complex_negative_amount(self) -> None:
        """Test create_test_service with 'complex' type and negative amount."""
        service_class = FlextTestsFactories.create_test_service("complex")
        service = service_class(amount=-10)

        result = service.execute()
        assert result.is_failure
        assert "Amount must be non-negative" in result.error

    def test_create_test_service_complex_disabled_with_amount(self) -> None:
        """Test create_test_service with 'complex' type disabled with amount."""
        service_class = FlextTestsFactories.create_test_service("complex")
        service = service_class(enabled=False, amount=100)

        result = service.execute()
        assert result.is_failure
        assert "Cannot have amount when disabled" in result.error

    def test_create_test_service_validate_business_rules_complex_valid(self) -> None:
        """Test validate_business_rules for complex service with valid data."""
        service_class = FlextTestsFactories.create_test_service("complex")
        service = service_class(name="Test", amount=100, enabled=True)

        result = service.validate_business_rules()
        assert result.is_success
        assert result.value is True

    def test_create_test_service_validate_business_rules_complex_invalid(self) -> None:
        """Test validate_business_rules for complex service with invalid data."""
        service_class = FlextTestsFactories.create_test_service("complex")
        service = service_class(name="")

        result = service.validate_business_rules()
        assert result.is_failure
        assert "Name is required" in result.error

    def test_create_test_service_validate_config_complex_valid(self) -> None:
        """Test validate_config for complex service with valid data."""
        service_class = FlextTestsFactories.create_test_service("complex")
        service = service_class(name="Test", amount=100)

        result = service.validate_config()
        assert result.is_success
        assert result.value is True

    def test_create_test_service_validate_config_name_too_long(self) -> None:
        """Test validate_config for complex service with name too long."""
        service_class = FlextTestsFactories.create_test_service("complex")
        long_name = "a" * 51
        service = service_class(name=long_name)

        result = service.validate_config()
        assert result.is_failure
        assert "Name too long" in result.error

    def test_create_test_service_validate_config_amount_too_large(self) -> None:
        """Test validate_config for complex service with amount too large."""
        service_class = FlextTestsFactories.create_test_service("complex")
        service = service_class(amount=1001)

        result = service.validate_config()
        assert result.is_failure
        assert "Value too large" in result.error

    def test_create_test_service_validate_config_non_complex(self) -> None:
        """Test validate_config for non-complex service."""
        service_class = FlextTestsFactories.create_test_service("test")
        service = service_class()

        result = service.validate_config()
        assert result.is_success
        assert result.value is True

    def test_create_test_service_validate_business_rules_non_complex(self) -> None:
        """Test validate_business_rules for non-complex service."""
        service_class = FlextTestsFactories.create_test_service("test")
        service = service_class()

        result = service.validate_business_rules()
        # Should call super() which returns success
        assert isinstance(result, FlextResult)
