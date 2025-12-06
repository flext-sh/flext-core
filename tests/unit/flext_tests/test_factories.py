"""Unit tests for flext_tests.factories module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import pytest

from flext_core import FlextResult
from flext_tests.factories import FlextTestsFactories, tt
from flext_tests.models import m

# Access models from centralized m.Tests.Factory namespace
User = m.Tests.Factory.User
Config = m.Tests.Factory.Config
Service = m.Tests.Factory.Service
Entity = m.Tests.Factory.Entity
ValueObject = m.Tests.Factory.ValueObject


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


# =============================================================================
# PHASE 7: Tests for Unified Factory Methods (tt.model, tt.res, tt.list, tt.dict, tt.generic)
# =============================================================================
class TestsFlextTestsFactoriesModel:
    """Tests for tt.model() unified method."""

    def test_model_user_default(self) -> None:
        """Test user model creation with defaults."""
        user = tt.model("user")
        assert isinstance(user, User)
        assert user.id is not None
        assert user.name == "Test User"
        assert "@example.com" in user.email
        assert user.active is True

    def test_model_user_custom(self) -> None:
        """Test user model creation with custom parameters."""
        user = tt.model("user", name="Custom User", email="custom@test.com")
        assert isinstance(user, User)
        assert user.name == "Custom User"
        assert user.email == "custom@test.com"

    def test_model_batch(self) -> None:
        """Test batch model creation."""
        users = tt.model("user", count=5)
        assert isinstance(users, list)
        assert len(users) == 5
        assert all(isinstance(user, User) for user in users)

    def test_model_as_result(self) -> None:
        """Test model wrapped in FlextResult."""
        result = tt.model("user", as_result=True)
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert isinstance(result.value, User)

    def test_model_as_dict(self) -> None:
        """Test model returned as dict."""
        users_dict = tt.model("user", count=3, as_dict=True)
        assert isinstance(users_dict, dict)
        assert len(users_dict) == 3
        assert all(isinstance(v, User) for v in users_dict.values())

    def test_model_config(self) -> None:
        """Test config model creation."""
        config = tt.model("config", environment="production")
        assert isinstance(config, Config)
        assert config.environment == "production"

    def test_model_service(self) -> None:
        """Test service model creation."""
        service = tt.model("service", service_type="database")
        assert isinstance(service, Service)
        assert service.type == "database"

    def test_model_entity(self) -> None:
        """Test entity model creation."""
        entity = tt.model("entity", name="Test Entity", value=42)
        assert isinstance(entity, Entity)
        assert entity.name == "Test Entity"

    def test_model_value_object(self) -> None:
        """Test value object model creation."""
        value_obj = tt.model("value", data="test_data", value_count=3)
        assert isinstance(value_obj, ValueObject)
        assert value_obj.data == "test_data"

    def test_model_with_transform(self) -> None:
        """Test model creation with transform function."""
        user = tt.model(
            "user",
            name="Original",
            transform=lambda u: User(
                id=u.id,
                name="Transformed",
                email=u.email,
                active=u.active,
            ),
        )
        assert user.name == "Transformed"

    def test_model_with_validate(self) -> None:
        """Test model creation with validation."""
        # Valid user (active=True)
        user = tt.model("user", active=True, validate=lambda u: u.active)
        assert user.active is True

        # Invalid user (active=False) - should return failure result
        result = tt.model(
            "user",
            active=False,
            validate=lambda u: u.active,
            as_result=True,
        )
        assert isinstance(result, FlextResult)
        assert result.is_failure


class TestsFlextTestsFactoriesRes:
    """Tests for tt.res() unified method."""

    def test_res_ok(self) -> None:
        """Test successful result creation."""
        result = tt.res("ok", value=42)
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert result.value == 42

    def test_res_fail(self) -> None:
        """Test failed result creation."""
        result = tt.res("fail", error="Error message")
        assert isinstance(result, FlextResult)
        assert result.is_failure
        assert result.error == "Error message"

    def test_res_fail_with_code(self) -> None:
        """Test failed result creation with error code."""
        result = tt.res("fail", error="Error message", error_code="ERR001")
        assert result.is_failure
        assert result.error == "Error message"
        # Note: error_code may be stored in result metadata

    def test_res_from_value_success(self) -> None:
        """Test from_value with non-None value."""
        result = tt.res("from_value", value=42)
        assert result.is_success
        assert result.value == 42

    def test_res_from_value_none(self) -> None:
        """Test from_value with None value."""
        result = tt.res("from_value", value=None, error_on_none="Value is required")
        assert result.is_failure
        assert "required" in result.error.lower()

    def test_res_batch_values(self) -> None:
        """Test batch result creation from values."""
        results = tt.res("ok", values=[1, 2, 3])
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(r.is_success for r in results)
        assert [r.value for r in results] == [1, 2, 3]

    def test_res_batch_errors(self) -> None:
        """Test batch result creation from errors."""
        results = tt.res("fail", errors=["err1", "err2"])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(r.is_failure for r in results)

    def test_res_mix_pattern(self) -> None:
        """Test batch result creation with mix pattern."""
        results = tt.res(
            "ok",
            values=[1, 2],
            errors=["e1", "e2"],
            mix_pattern=[True, False, True, False],
        )
        assert len(results) == 4
        assert results[0].is_success and results[0].value == 1
        assert results[1].is_failure
        assert results[2].is_success and results[2].value == 2
        assert results[3].is_failure

    def test_res_with_transform(self) -> None:
        """Test result creation with transform function."""
        result = tt.res("ok", value=5, transform=lambda x: x * 2)
        assert result.is_success
        assert result.value == 10


class TestsFlextTestsFactoriesList:
    """Tests for tt.list() method."""

    def test_list_from_model(self) -> None:
        """Test list creation from model kind."""
        users = tt.list("user", count=3)
        assert isinstance(users, list)
        assert len(users) == 3
        assert all(isinstance(u, User) for u in users)

    def test_list_from_callable(self) -> None:
        """Test list creation from callable factory."""
        numbers = tt.list(lambda: 42, count=5)
        assert numbers == [42, 42, 42, 42, 42]

    def test_list_from_sequence(self) -> None:
        """Test list creation from sequence."""
        doubled = tt.list([1, 2, 3], transform=lambda x: x * 2)
        assert doubled == [2, 4, 6]

    def test_list_with_filter(self) -> None:
        """Test list creation with filter."""
        evens = tt.list([1, 2, 3, 4, 5], filter_=lambda x: x % 2 == 0)
        assert evens == [2, 4]

    def test_list_with_unique(self) -> None:
        """Test list creation with uniqueness."""
        # Create list with duplicates
        items = tt.list([1, 2, 2, 3, 3, 3], unique=True)
        assert len(items) == 3
        assert set(items) == {1, 2, 3}

    def test_list_as_result(self) -> None:
        """Test list creation wrapped in result."""
        result = tt.list("user", count=3, as_result=True)
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert len(result.value) == 3


class TestsFlextTestsFactoriesDict:
    """Tests for tt.dict_factory() method."""

    def test_dict_from_model(self) -> None:
        """Test dict creation from model kind."""
        users = tt.dict_factory("user", count=3)
        assert isinstance(users, dict)
        assert len(users) == 3
        assert all(isinstance(u, User) for u in users.values())

    def test_dict_with_key_factory(self) -> None:
        """Test dict creation with key factory."""
        users = tt.dict_factory("user", count=3, key_factory=lambda i: f"user_{i}")
        assert set(users.keys()) == {"user_0", "user_1", "user_2"}

    def test_dict_with_value_factory(self) -> None:
        """Test dict creation with value factory."""

        def value_factory(key: str) -> User:
            return User(id=key, name=f"User {key}", email=f"{key}@test.com")

        users = tt.dict_factory("user", count=2, value_factory=value_factory)
        assert len(users) == 2

    def test_dict_from_mapping(self) -> None:
        """Test dict creation from existing mapping."""
        existing = {"a": 1, "b": 2}
        merged = tt.dict_factory(existing, merge_with={"c": 3})
        assert merged == {"a": 1, "b": 2, "c": 3}

    def test_dict_as_result(self) -> None:
        """Test dict creation wrapped in result."""
        result = tt.dict_factory("user", count=3, as_result=True)
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert len(result.value) == 3


class TestsFlextTestsFactoriesGeneric:
    """Tests for tt.generic() method."""

    def test_generic_simple(self) -> None:
        """Test generic type instantiation."""

        class SimpleClass:
            def __init__(self, name: str) -> None:
                self.name = name

        obj = tt.generic(SimpleClass, kwargs={"name": "test"})
        assert isinstance(obj, SimpleClass)
        assert obj.name == "test"

    def test_generic_with_args(self) -> None:
        """Test generic type instantiation with positional args."""

        class ArgsClass:
            def __init__(self, a: int, b: int, c: str = "default") -> None:
                self.a = a
                self.b = b
                self.c = c

        obj = tt.generic(ArgsClass, args=[1, 2], kwargs={"c": "custom"})
        assert obj.a == 1
        assert obj.b == 2
        assert obj.c == "custom"

    def test_generic_batch(self) -> None:
        """Test batch generic type instantiation."""

        class BatchClass:
            def __init__(self, value: int) -> None:
                self.value = value

        objs = tt.generic(BatchClass, kwargs={"value": 42}, count=5)
        assert isinstance(objs, list)
        assert len(objs) == 5
        assert all(isinstance(o, BatchClass) for o in objs)
        assert all(o.value == 42 for o in objs)

    def test_generic_with_validate(self) -> None:
        """Test generic type instantiation with validation."""

        class ValidatedClass:
            def __init__(self, age: int) -> None:
                self.age = age

        # Valid age
        obj = tt.generic(
            ValidatedClass,
            kwargs={"age": 25},
            validate=lambda o: o.age >= 18,
        )
        assert obj.age == 25

        # Invalid age - should raise ValueError
        with pytest.raises(ValueError, match="Validation failed"):
            tt.generic(
                ValidatedClass,
                kwargs={"age": 15},
                validate=lambda o: o.age >= 18,
            )

    def test_generic_as_result(self) -> None:
        """Test generic type instantiation wrapped in result."""

        class ResultClass:
            def __init__(self, value: str) -> None:
                self.value = value

        result = tt.generic(ResultClass, kwargs={"value": "test"}, as_result=True)
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert isinstance(result.value, ResultClass)
        assert result.value.value == "test"
